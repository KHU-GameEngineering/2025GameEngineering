import open3d as o3d
import trimesh
import numpy as np
import os
from scipy.spatial import cKDTree

# --- 변환 헬퍼 함수 ---
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

# --- 수리 함수 ---
def ultimate_repair(mesh_trimesh):
    print("   -> [Repair] 갈라진 틈 봉합 및 수리...")
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

# --- 복셀 리메싱 ---
def voxel_remesh_cleanup(mesh_o3d):
    if len(mesh_o3d.triangles) == 0: return mesh_o3d
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    actual_voxel_size = max(bbox.get_extent()) * 0.003 
    return mesh_o3d.simplify_vertex_clustering(voxel_size=actual_voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)

# --- 색상 강제 전사 (Reprojection) ---
def reproject_colors(target_mesh_o3d, source_mesh_trimesh):
    print("\n   -> [Color] 원본 색상 강제 전사 중 (Reprojection)...")
    
    # 1. 원본 색상 데이터 추출
    if hasattr(source_mesh_trimesh.visual, 'vertex_colors') and len(source_mesh_trimesh.visual.vertex_colors) > 0:
        source_colors = source_mesh_trimesh.visual.vertex_colors[:, :3] # Alpha 제거
    else:
        print("      [경고] 원본 파일에서 Vertex Color를 찾을 수 없습니다!")
        return target_mesh_o3d

    # 2. KDTree 및 매핑
    tree = cKDTree(source_mesh_trimesh.vertices)
    target_vertices = np.asarray(target_mesh_o3d.vertices)
    dists, indices = tree.query(target_vertices)
    
    new_colors = source_colors[indices] / 255.0
    target_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    print("      [성공] 색상 전사 완료.")
    
    return target_mesh_o3d

def process_ultra_smooth_color(input_path, output_folder, target_low_count=3000):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 로드 (원본 보존용 & 작업용)
    print(f"STEP 1: 로드... [{os.path.basename(input_path)}]")
    try:
        # 색상 복구용 원본 (수리 안 함)
        original_trimesh = trimesh.load(input_path, file_type='obj', force='mesh')
        
        # 작업용 복사본
        work_trimesh = trimesh.load(input_path, file_type='obj', force='mesh')
    except Exception as e:
        print(f"[로딩 오류] {e}")
        return

    work_trimesh = ultimate_repair(work_trimesh)
    mesh_o3d = trimesh_to_o3d(work_trimesh)
    mesh_o3d = voxel_remesh_cleanup(mesh_o3d) 

    # 2. 다운샘플링 (High -> Low)
    print(f"\nSTEP 2: 베이스 최적화 (Target: {target_low_count})...")
    mesh_low_o3d = mesh_o3d.simplify_quadric_decimation(
        target_number_of_triangles=target_low_count
    )
    mesh_low_trimesh = o3d_to_trimesh(mesh_low_o3d)
    mesh_low_trimesh = ultimate_repair(mesh_low_trimesh) 
    
    # 3. 업샘플링 (Loop Subdivision x2)
    print(f"\nSTEP 3: Loop Subdivision (x16)...")
    mesh_high_trimesh = mesh_low_trimesh.subdivide() 
    mesh_high_trimesh = mesh_high_trimesh.subdivide()
    mesh_high_o3d = trimesh_to_o3d(mesh_high_trimesh)

    # 4. 하이브리드 스무딩 (Taubin + Laplacian)
    print(f"\nSTEP 4: 하이브리드 스무딩 (Taubin + Laplacian)...")
    
    if len(mesh_high_o3d.triangles) > 0:
        print("   -> 1차: Taubin Smoothing (x50) - 형태 잡기")
        mesh_final = mesh_high_o3d.filter_smooth_taubin(number_of_iterations=50)
        
        print("   -> 2차: Laplacian Smoothing (x8) - 모서리 녹이기")
        mesh_final = mesh_final.filter_smooth_simple(number_of_iterations=8)
    else:
        mesh_final = mesh_high_o3d
    
    if len(mesh_final.triangles) > 60000:
         mesh_final = mesh_final.simplify_quadric_decimation(target_number_of_triangles=60000)

    # 5. 색상 강제 전사 (Reprojection)
    mesh_final = reproject_colors(target_mesh_o3d=mesh_final, source_mesh_trimesh=original_trimesh)

    # 법선 재계산
    mesh_final.compute_vertex_normals()
    
    # 저장
    save_path = os.path.abspath(os.path.join(output_folder, "LOD_High_UltraSmooth_Color.obj"))
    
    # write_vertex_colors=True 필수
    if o3d.io.write_triangle_mesh(save_path, mesh_final, write_vertex_normals=True, write_vertex_colors=True):
        print(f"\n[성공] 변환 완료!")
        print(f" - 결과물: {save_path}")
        print(f" - 최종 폴리곤: {len(mesh_final.triangles):,}개")
    else:
        print("[오류] 파일 저장 실패.")

if __name__ == "__main__":
    INPUT_FILE = "instant_mesh_raw.obj"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "input", INPUT_FILE)
    output_dir = os.path.join(current_dir, "output_ultra_color")
    
    if os.path.exists(input_path):
        process_ultra_smooth_color(input_path, output_dir, target_low_count=3000)
    else:
        print("[오류] 입력 파일 없음")