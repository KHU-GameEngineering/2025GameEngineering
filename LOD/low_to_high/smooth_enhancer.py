import open3d as o3d
import trimesh
import numpy as np
import os

# --- [공통] 변환 헬퍼 함수 ---
def o3d_to_trimesh(mesh_o3d):
    return trimesh.Trimesh(np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles), process=False)

def trimesh_to_o3d(mesh_trimesh):
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_trimesh.vertices), o3d.utility.Vector3iVector(mesh_trimesh.faces))

# --- [공통] 초강력 수리 함수 (Ultimate Repair) ---
def ultimate_repair(mesh_trimesh):
    print("   -> [Repair] 갈라진 틈 봉합 및 지오메트리 수리...")
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

# --- [High 전용] 복셀 리메싱 ---
def voxel_remesh_cleanup(mesh_o3d):
    if len(mesh_o3d.triangles) == 0: return mesh_o3d
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    actual_voxel_size = max(bbox.get_extent()) * 0.003 
    return mesh_o3d.simplify_vertex_clustering(voxel_size=actual_voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)

def process_perfect_smooth(input_path, output_folder, base_target=3000):
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 1. 로드 & 수리
    print(f"STEP 1: 로드 및 수리... [{os.path.basename(input_path)}]")
    try:
        mesh_trimesh = trimesh.load(input_path, file_type='obj', force='mesh')
        mesh_trimesh = ultimate_repair(mesh_trimesh)
        mesh_o3d = trimesh_to_o3d(mesh_trimesh)
        mesh_o3d = voxel_remesh_cleanup(mesh_o3d) # 토폴로지 청소
    except Exception as e:
        print(f"[오류] {e}")
        return

    # 2. 다운샘플링 (베이스 생성)
    print(f"\nSTEP 2: 베이스 최적화 (Target: {base_target})...")
    mesh_low = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=base_target)
    
    # 2차 수리
    mesh_low_trimesh = o3d_to_trimesh(mesh_low)
    mesh_low_trimesh = ultimate_repair(mesh_low_trimesh)
    
    # Low 저장
    low_save_path = os.path.abspath(os.path.join(output_folder, "Base_Low.obj"))
    o3d.io.write_triangle_mesh(low_save_path, trimesh_to_o3d(mesh_low_trimesh))
    print(f" -> Base Low 저장 완료 ({len(mesh_low_trimesh.faces)} tris)")

    # 3. 업샘플링 (Loop Subdivision)
    print(f"\nSTEP 3: Loop Subdivision (x16)...")
    mesh_high_trimesh = mesh_low_trimesh.subdivide()
    mesh_high_trimesh = mesh_high_trimesh.subdivide()
    mesh_high = trimesh_to_o3d(mesh_high_trimesh)

    # 4. 후처리 (Taubin Smoothing)
    print(f"\nSTEP 4: Taubin Smoothing (x30)...")
    if len(mesh_high.triangles) > 0:
        mesh_final = mesh_high.filter_smooth_taubin(number_of_iterations=30)
    else: mesh_final = mesh_high

    if len(mesh_final.triangles) > 60000:
         mesh_final = mesh_final.simplify_quadric_decimation(target_number_of_triangles=60000)

    mesh_final.compute_vertex_normals() # 쉐이딩

    # 저장
    save_path = os.path.abspath(os.path.join(output_folder, "Final_Smooth_High.obj"))
    o3d.io.write_triangle_mesh(save_path, mesh_final, write_vertex_normals=True)
    print(f"\n[성공] 최종 완료! ({len(mesh_final.triangles):,} tris)")

if __name__ == "__main__":
    INPUT_FILE = "instant_mesh_raw.obj"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "input", INPUT_FILE)
    output_dir = os.path.join(current_dir, "output_smooth")
    
    if os.path.exists(input_path):
        process_perfect_smooth(input_path, output_dir, base_target=3000)
    else: print("[오류] 입력 파일 없음")