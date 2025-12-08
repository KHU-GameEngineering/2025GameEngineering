import open3d as o3d
import trimesh
import numpy as np
import os
from scipy.spatial import cKDTree

# --- 기하학적 변환 헬퍼 ---
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

# --- 위상 정규화 (Topology Regularization) ---
def isotropic_remeshing_approximation(mesh_trimesh, target_len_percent=0.01):
    """
    [Isotropic Remeshing]
    불규칙한 삼각형(Anisotropic)을 정삼각형(Isotropic)에 가깝게 재배열합니다.
    Subdivision의 수학적 안정성을 보장하기 위한 필수 전처리 단계입니다.
    """
    print("   -> [Topology] 등방성 리메싱 (Isotropic Approximation)...")
    
    # 1. 평균 엣지 길이 분석 (Discrete Differential Geometry)
    edges = mesh_trimesh.edges_unique_length
    mean_edge = np.mean(edges)
    
    # 2. 목표 엣지 길이 설정 (너무 긴 엣지는 자르고, 짧은 엣지는 놔둠)
    # 전체 스케일의 1% 정도 길이로 균일화 시도
    target_length = mesh_trimesh.scale * target_len_percent
    
    # 3. 긴 엣지 절단 (Subdivide long edges)
    # 한 번에 너무 많이 자르면 느리므로 평균보다 2배 긴 것들만 우선 처리
    if mean_edge > target_length * 2:
        new_mesh = mesh_trimesh.subdivide_to_size(target_length * 1.5)
        return new_mesh
    
    return mesh_trimesh

# ---  수리 및 매니폴드 복원 ---
def manifold_repair(mesh_trimesh):
    print("   -> [Repair] 비-매니폴드 구조 수리 및 병합...")
    # Geometry Welding: 위상적으로 떨어진 정점을 유클리드 거리 기준으로 병합
    mesh_trimesh.merge_vertices(merge_tex=False, merge_norm=False)
    
    # Degenerate Elements Removal
    try:
        mesh_trimesh.update_faces(mesh_trimesh.nondegenerate_faces)
        mesh_trimesh.update_faces(mesh_trimesh.unique_faces)
    except: pass
    
    # Hole Filling (Watertightness)
    try: trimesh.repair.fill_holes(mesh_trimesh)
    except: pass
    
    # Component Filtering (Noise Removal)
    components = mesh_trimesh.split(only_watertight=False)
    if len(components) > 1:
        total_area = sum(c.area for c in components)
        # 전체 면적의 0.5% 미만인 가우시안 노이즈(먼지) 제거
        valid = [c for c in components if c.area > total_area * 0.005 or len(c.faces) > 50]
        if valid: mesh_trimesh = trimesh.util.concatenate(valid)
        else: mesh_trimesh = max(components, key=lambda m: len(m.faces))

    trimesh.repair.fix_normals(mesh_trimesh)
    return mesh_trimesh

# --- [학술적 모듈 4] 신호 복원 (Signal Reconstruction) ---
def transfer_color_signal(target_mesh_o3d, source_mesh_trimesh):
    """
    [Attribute Transfer]
    Nearest Neighbor Search(KD-Tree)를 이용해 원본의 색상 신호를 
    고해상도 메쉬로 전사(Reprojection)합니다.
    """
    print("\n   -> [Signal] 색상 신호 복원 (Color Reprojection)...")
    
    if hasattr(source_mesh_trimesh.visual, 'vertex_colors') and len(source_mesh_trimesh.visual.vertex_colors) > 0:
        source_colors = source_mesh_trimesh.visual.vertex_colors[:, :3]
    else:
        print("      [Warning] 원본 신호 부재: Vertex Color 없음.")
        return target_mesh_o3d

    # KD-Tree 구축 (Spatial Indexing)
    tree = cKDTree(source_mesh_trimesh.vertices)
    target_vertices = np.asarray(target_mesh_o3d.vertices)
    
    # 쿼리 (Query)
    dists, indices = tree.query(target_vertices)
    
    # 신호 매핑 (Mapping)
    new_colors = source_colors[indices] / 255.0
    target_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    
    return target_mesh_o3d

# --- 메인 파이프라인 ---
def process_academic_upsampling(input_path, output_folder, target_base_poly=4000):
    
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 1. 원본 데이터 로드 (Source Signal)
    print(f"STEP 1: 데이터 로드... [{os.path.basename(input_path)}]")
    try:
        source_mesh = trimesh.load(input_path, file_type='obj', force='mesh')
        work_mesh = trimesh.load(input_path, file_type='obj', force='mesh')
    except Exception as e:
        print(f"[Critical Error] {e}")
        return

    # 2. 전처리: 위상 정규화 (Topology Regularization)
    # 기존 코드엔 없던 과정. 메쉬의 품질을 균일하게 만듦.
    print(f"\nSTEP 2: 위상 정규화 (Topology Regularization)...")
    work_mesh = manifold_repair(work_mesh)
    work_mesh = isotropic_remeshing_approximation(work_mesh) # <--- NEW: 등방성 리메싱

    # 3. 베이스 최적화 (Base Parameterization)
    # 리메싱된 고품질 메쉬를 기반으로 QEM 수행 -> 결과가 훨씬 예쁨
    mesh_o3d = trimesh_to_o3d(work_mesh)
    print(f"   -> QEM Decimation (Target: {target_base_poly})...")
    mesh_base = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_base_poly)
    
    # 수리
    mesh_base_trimesh = o3d_to_trimesh(mesh_base)
    mesh_base_trimesh = manifold_repair(mesh_base_trimesh)

    # 4. 고해상도 복원 (High-Res Reconstruction)
    # Loop Subdivision: C2 연속성을 가진 곡면 생성 알고리즘
    print(f"\nSTEP 3: 곡면 재구성 (Loop Subdivision Surface)...")
    mesh_high_trimesh = mesh_base_trimesh.subdivide()
    mesh_high_trimesh = mesh_high_trimesh.subdivide()
    mesh_high_o3d = trimesh_to_o3d(mesh_high_trimesh)

    # 5. 기하학적 스무딩 (Geometric Filtering)
    # Signal Processing Approach to Fair Surface Design (Taubin, 1995)
    # 부피 수축 없는 Low-pass Filtering 수행
    print(f"\nSTEP 4: 신호 필터링 (Taubin Smoothing)...")
    if len(mesh_high_o3d.triangles) > 0:
        # High-Frequency Noise 제거
        mesh_final = mesh_high_o3d.filter_smooth_taubin(number_of_iterations=50)
        
        # 미세 특징 복원 (Laplacian)
        # Feature-preserving을 위해 약하게 적용
        mesh_final = mesh_final.filter_smooth_simple(number_of_iterations=5)
    else:
        mesh_final = mesh_high_o3d

    # 용량 제약 조건 (Capacity Constraint)
    if len(mesh_final.triangles) > 60000:
         mesh_final = mesh_final.simplify_quadric_decimation(target_number_of_triangles=60000)

    # 6. 신호 전사 (Color Reconstruction)
    mesh_final = transfer_color_signal(mesh_final, source_mesh)

    # 7. 법선 재계산 (Shading Normal Reconstruction)
    mesh_final.compute_vertex_normals()
    
    # 저장
    save_path = os.path.abspath(os.path.join(output_folder, "Academic_HighPoly.obj"))
    if o3d.io.write_triangle_mesh(save_path, mesh_final, write_vertex_normals=True, write_vertex_colors=True):
        print(f"\n[Success] 학술적 파이프라인 처리 완료.")
        print(f" - Output: {save_path}")
        print(f" - Vertices: {len(mesh_final.vertices):,}")
        print(f" - Faces: {len(mesh_final.triangles):,}")
    else:
        print("[Error] 저장 실패 (권한 확인 필요)")

if __name__ == "__main__":
    INPUT_FILE = "instant_mesh_raw.obj"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "input", INPUT_FILE)
    output_dir = os.path.join(current_dir, "output_academic")
    
    if os.path.exists(input_path):
        process_academic_upsampling(input_path, output_dir, target_base_poly=4000)
    else:
        print("[오류] 입력 파일 없음")