import open3d as o3d
import trimesh
import numpy as np
import os
import time
import copy
from scipy.spatial import cKDTree

# ==============================================================================
# [MODULE 1] 공통 헬퍼 및 수리 함수 (Shared Utilities)
# ==============================================================================

def o3d_to_trimesh(mesh_o3d):
    return trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=False
    )

def trimesh_to_o3d(mesh_trimesh):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_trimesh.vertices),
        o3d.utility.Vector3iVector(mesh_trimesh.faces)
    )

def ultimate_repair(mesh_trimesh):
    """갈라진 틈 봉합 및 지오메트리 수리"""
    print("   -> [System] 메쉬 수리 및 병합 중...")
    mesh_trimesh.merge_vertices(merge_tex=False, merge_norm=False)
    try:
        mesh_trimesh.update_faces(mesh_trimesh.nondegenerate_faces)
        mesh_trimesh.update_faces(mesh_trimesh.unique_faces)
    except: pass
    try: trimesh.repair.fill_holes(mesh_trimesh)
    except: pass
    
    components = mesh_trimesh.split(only_watertight=False)
    if len(components) > 1:
        total_area = sum(c.area for c in components)
        valid = [c for c in components if c.area > total_area * 0.005 or len(c.faces) > 50]
        if valid: mesh_trimesh = trimesh.util.concatenate(valid)
        else: mesh_trimesh = max(components, key=lambda m: len(m.faces))
    
    trimesh.repair.fix_normals(mesh_trimesh)
    return mesh_trimesh

def transfer_color_signal(target_mesh_o3d, source_mesh):
    """색상 신호 전사 (Open3D mesh <-> Trimesh/Open3D Source)"""
    print("   -> [System] 색상 신호 복원 (Reprojection)...")
    
    # 소스 데이터 추출 (Open3D 객체인지 Trimesh 객체인지 확인)
    if isinstance(source_mesh, o3d.geometry.TriangleMesh):
        if not source_mesh.has_vertex_colors(): return target_mesh_o3d
        source_verts = np.asarray(source_mesh.vertices)
        source_colors = np.asarray(source_mesh.vertex_colors) # 0.0-1.0 float
    else: # Trimesh
        if not hasattr(source_mesh.visual, 'vertex_colors'): return target_mesh_o3d
        source_verts = source_mesh.vertices
        source_colors = source_mesh.visual.vertex_colors[:, :3] / 255.0 # 0-255 -> 0.0-1.0
        
    tree = cKDTree(source_verts)
    dists, indices = tree.query(np.asarray(target_mesh_o3d.vertices))
    
    new_colors = source_colors[indices]
    target_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    return target_mesh_o3d

# ==============================================================================
# [MODULE 2] 학술적 업샘플링 (Low -> High)
# ==============================================================================

def isotropic_remeshing(mesh_trimesh, target_len_percent=0.01):
    edges = mesh_trimesh.edges_unique_length
    mean_edge = np.mean(edges)
    target_length = mesh_trimesh.scale * target_len_percent
    if mean_edge > target_length * 2:
        return mesh_trimesh.subdivide_to_size(target_length * 1.5)
    return mesh_trimesh

def run_academic_upsampling(input_path, output_folder):
    print("\n=== [Mode] Low Poly -> High Poly Refinement ===")
    
    # 1. 로드
    try:
        source_mesh = trimesh.load(input_path, file_type='obj', force='mesh')
        work_mesh = trimesh.load(input_path, file_type='obj', force='mesh')
    except Exception as e:
        print(f"[Error] {e}")
        return

    # 2. 전처리
    print("Step 1: 위상 정규화...")
    work_mesh = ultimate_repair(work_mesh)
    work_mesh = isotropic_remeshing(work_mesh)

    # 3. 베이스 최적화
    print("Step 2: 베이스 파라미터화 (QEM)...")
    mesh_o3d = trimesh_to_o3d(work_mesh)
    mesh_base = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=4000)
    
    # 4. 업샘플링
    print("Step 3: 곡면 재구성 (Loop Subdivision)...")
    mesh_base_trimesh = o3d_to_trimesh(mesh_base)
    mesh_base_trimesh = ultimate_repair(mesh_base_trimesh)
    mesh_high = mesh_base_trimesh.subdivide().subdivide() # x16
    mesh_high_o3d = trimesh_to_o3d(mesh_high)

    # 5. 스무딩
    print("Step 4: 신호 필터링 (Taubin + Laplacian)...")
    if len(mesh_high_o3d.triangles) > 0:
        mesh_final = mesh_high_o3d.filter_smooth_taubin(number_of_iterations=50)
        mesh_final = mesh_final.filter_smooth_simple(number_of_iterations=5)
    else:
        mesh_final = mesh_high_o3d
    
    if len(mesh_final.triangles) > 60000:
        mesh_final = mesh_final.simplify_quadric_decimation(target_number_of_triangles=60000)

    # 6. 색상 복원
    mesh_final = transfer_color_signal(mesh_final, source_mesh)
    mesh_final.compute_vertex_normals()

    # 저장
    save_path = os.path.join(output_folder, "Refined_HighPoly.obj")
    o3d.io.write_triangle_mesh(save_path, mesh_final, write_vertex_normals=True, write_vertex_colors=True)
    print(f"[완료] 저장됨: {save_path}")

# ==============================================================================
# [MODULE 3] 스마트 다운샘플링 (High -> Low)
# ==============================================================================

def run_smart_decimation(input_path, output_folder, mode_selection):
    print("\n=== [Mode] High Poly -> Low Poly Optimization ===")
    
    mesh = o3d.io.read_triangle_mesh(input_path)
    if not mesh.has_triangles(): return
    
    # 기본 정리
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()
    
    original_count = len(mesh.triangles)
    print(f" -> 원본 폴리곤: {original_count:,}개")
    
    # 목표 설정
    targets = {
        "High": min(original_count, 50000),
        "Mid":  min(int(original_count * 0.2), 10000),
        "Low":  min(max(int(original_count * 0.02), 500), 2000)
    }
    
    # 사용자 선택 처리
    tasks = []
    if mode_selection == "ALL":
        tasks = [("LOD0_High", targets["High"]), ("LOD1_Mid", targets["Mid"]), ("LOD2_Low", targets["Low"])]
    elif mode_selection == "HIGH":
        tasks = [("LOD0_High", targets["High"])]
    elif mode_selection == "MID":
        tasks = [("LOD1_Mid", targets["Mid"])]
    elif mode_selection == "LOW":
        tasks = [("LOD2_Low", targets["Low"])]
    
    for label, count in tasks:
        print(f"\nProcessing {label} (Target: {count})...")
        
        # 1. Decimation
        if original_count > count:
            mesh_lod = mesh.simplify_quadric_decimation(target_number_of_triangles=count)
        else:
            mesh_lod = copy.deepcopy(mesh)
            
        # 2. Color Transfer
        if mesh.has_vertex_colors():
            mesh_lod = transfer_color_signal(mesh_lod, mesh)
            
        # 3. Styling
        if "Low" in label:
            mesh_lod.vertex_normals = o3d.utility.Vector3dVector([])
        else:
            mesh_lod.compute_vertex_normals()
            
        save_path = os.path.join(output_folder, f"{label}.obj")
        o3d.io.write_triangle_mesh(save_path, mesh_lod, write_vertex_normals=True, write_vertex_colors=True)
        print(f" -> 저장됨: {save_path}")

# ==============================================================================
# [MODULE 4] 단순 정리 (Simple Clean)
# ==============================================================================

def run_simple_clean(input_path, output_folder):
    print("\n=== [Mode] Low Poly Raw Mesh Cleanup ===")
    mesh = trimesh.load(input_path, file_type='obj', force='mesh')
    mesh = ultimate_repair(mesh)
    
    save_path = os.path.join(output_folder, "Cleaned_LowPoly.obj")
    mesh.export(save_path)
    print(f"[완료] 저장됨: {save_path}")
