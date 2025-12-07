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

# --- 수리 함수 (기하학적 수리에 집중) ---
def ultimate_repair(mesh_trimesh):
    print("   -> [Repair] 갈라진 틈 봉합 (Geometry Weld)...")
    # 색상 신경 쓰지 말고 일단 모양부터 하나로 합침
    mesh_trimesh.merge_vertices(merge_tex=False, merge_norm=False)
    
    try:
        mesh_trimesh.update_faces(mesh_trimesh.nondegenerate_faces)
        mesh_trimesh.update_faces(mesh_trimesh.unique_faces)
    except: pass
    try: trimesh.repair.fill_holes(mesh_trimesh)
    except: pass

    # 파편 제거
    components = mesh_trimesh.split(only_watertight=False)
    if len(components) > 1:
        total_area = sum(c.area for c in components)
        valid = [c for c in components if c.area > total_area * 0.005 or len(c.faces) > 50]
        if valid: mesh_trimesh = trimesh.util.concatenate(valid)
        else: mesh_trimesh = max(components, key=lambda m: len(m.faces))

    trimesh.repair.fix_normals(mesh_trimesh)
    return mesh_trimesh

# --- [핵심 함수] 색상 강제 전사 (Reprojection) ---
def reproject_colors(target_mesh_o3d, source_mesh_trimesh):
    """
    원본(Source)의 색상을 타겟(Target)에 강제로 입히는 함수
    """
    print("\n   -> [Color] 원본 색상 강제 전사 중 (Reprojection)...")
    
    # 1. 원본 색상 데이터 추출
    # Trimesh 로드 시 색상이 visual.vertex_colors에 있음 (RGBA or RGB)
    if hasattr(source_mesh_trimesh.visual, 'vertex_colors') and len(source_mesh_trimesh.visual.vertex_colors) > 0:
        source_colors = source_mesh_trimesh.visual.vertex_colors[:, :3] # Alpha 제거하고 RGB만
    else:
        print("      [경고] 원본 파일에서 Vertex Color를 찾을 수 없습니다!")
        return target_mesh_o3d

    # 2. KDTree 생성 (원본 정점 위치 기준)
    tree = cKDTree(source_mesh_trimesh.vertices)
    
    # 3. 타겟 정점 위치 준비
    target_vertices = np.asarray(target_mesh_o3d.vertices)
    
    # 4. 가장 가까운 점 찾기
    # 타겟의 각 점에 대해, 원본에서 가장 가까운 점의 인덱스를 찾음
    dists, indices = tree.query(target_vertices)
    
    # 5. 색상 매핑
    # 원본(0~255 uint8) -> Open3D(0.0~1.0 float) 변환
    new_colors = source_colors[indices] / 255.0
    
    # 6. 적용
    target_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    print("      [성공] 색상 전사 완료.")
    
    return target_mesh_o3d


def process_high_poly_with_color(input_path, output_folder, target_low=3000):
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 1. 로드 (원본 보존용 & 변환용)
    print(f"STEP 1: 로드... [{os.path.basename(input_path)}]")
    try:
        # force='mesh'로 로드하여 색상 데이터 확보
        original_trimesh = trimesh.load(input_path, file_type='obj', force='mesh')
        
        # 변환 작업을 위한 복사본
        work_trimesh = trimesh.load(input_path, file_type='obj', force='mesh')
    except Exception as e:
        print(f"[오류] {e}")
        return

    # ----------------------------------------------------
    # PHASE I: 수리 및 다운샘플링
    # ----------------------------------------------------
    work_trimesh = ultimate_repair(work_trimesh)
    mesh_o3d = trimesh_to_o3d(work_trimesh)

    print(f"\nSTEP 2: 베이스 최적화 (Target: {target_low})...")
    # 여기서 색상이 날아가든 말든 신경 쓰지 않습니다. (나중에 입힐 거니까)
    mesh_low = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_low)
    
    # Low 수리
    mesh_low_trimesh = o3d_to_trimesh(mesh_low)
    mesh_low_trimesh = ultimate_repair(mesh_low_trimesh)

    # ----------------------------------------------------
    # PHASE II: 업샘플링 (Subdivision)
    # ----------------------------------------------------
    print(f"\nSTEP 3: Loop Subdivision (x16)...")
    # Subdivision 수행 (이때 색상 정보가 깨지거나 사라질 수 있음)
    mesh_high_trimesh = mesh_low_trimesh.subdivide()
    mesh_high_trimesh = mesh_high_trimesh.subdivide()
    
    mesh_high_o3d = trimesh_to_o3d(mesh_high_trimesh)

    # ----------------------------------------------------
    # PHASE III: 스무딩 (Taubin)
    # ----------------------------------------------------
    print(f"\nSTEP 4: Taubin Smoothing (x30)...")
    if len(mesh_high_o3d.triangles) > 0:
        mesh_final = mesh_high_o3d.filter_smooth_taubin(number_of_iterations=30)
    else:
        mesh_final = mesh_high_o3d

    if len(mesh_final.triangles) > 60000:
         mesh_final = mesh_final.simplify_quadric_decimation(target_number_of_triangles=60000)

    # ----------------------------------------------------
    # [핵심 단계] PHASE IV: 색상 복구 (Color Reprojection)
    # 모양이 다 잡힌 High Poly에 원본의 색을 다시 칠합니다.
    # ----------------------------------------------------
    mesh_final = reproject_colors(target_mesh_o3d=mesh_final, source_mesh_trimesh=original_trimesh)

    # 법선 계산
    mesh_final.compute_vertex_normals()

    # 저장
    save_path = os.path.abspath(os.path.join(output_folder, "LOD_High_Colored.obj"))
    
    # write_vertex_colors=True 필수
    if o3d.io.write_triangle_mesh(save_path, mesh_final, write_vertex_normals=True, write_vertex_colors=True):
        print(f"\n[성공] 변환 완료!")
        print(f" - 결과물: {save_path}")
        print(f" - 폴리곤: {len(mesh_final.triangles):,}개")
    else:
        print("[실패] 저장 권한 확인 필요")

if __name__ == "__main__":
    INPUT_FILE = "instant_mesh_raw.obj"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "input", INPUT_FILE)
    output_dir = os.path.join(current_dir, "output_high_color")
    
    if os.path.exists(input_path):
        process_high_poly_with_color(input_path, output_dir, target_low=3000)
    else:
        print("[오류] 파일 없음")