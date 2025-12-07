import open3d as o3d
import os
import time
import copy

# LOD 단계 정의
LOD_ORDER = ["LOD0_High", "LOD1_Mid", "LOD2_Low"]

def calculate_target_counts(original_count):
    """
    [스마트] 원본 폴리곤 수에 따라 LOD 목표치를 유동적으로 계산
    """
    targets = {}
    
    # LOD 0: 원본 유지 (최대 5만)
    targets["LOD0_High"] = min(original_count, 50000)
    
    # LOD 1: 원본의 20% (최대 1만)
    mid_target = int(original_count * 0.2)
    targets["LOD1_Mid"] = min(mid_target, 10000)
    
    # LOD 2: 원본의 2% (최소 500 ~ 최대 2000)
    low_target = int(original_count * 0.02)
    low_target = max(low_target, 500) 
    targets["LOD2_Low"] = min(low_target, 2000) 
    
    return targets

def simple_cleanup(mesh):
    """
    [롤백됨] 강력한 수리 대신 기본적인 정리만 수행
    구조를 강제로 합치지 않으므로 원본의 결이 유지됩니다.
    """
    print(" -> 기본 메쉬 정리 중 (중복 제거)...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh

def process_smart_lod(input_path, output_folder):
    # 0. 준비
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\nStep 1: 모델 로드 중... [{os.path.basename(input_path)}]")
    # Open3D로 바로 로드 (Trimesh 안 씀)
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if not mesh.has_triangles():
        print("[오류] 메쉬 정보를 읽을 수 없습니다.")
        return
    
    # 1. 가벼운 정리 (Repair 제거됨)
    mesh = simple_cleanup(mesh)
    mesh.compute_vertex_normals() 

    original_count = len(mesh.triangles)
    print(f" -> 원본 폴리곤 개수: {original_count:,}개")

    # 2. 목표치 계산
    lod_targets = calculate_target_counts(original_count)
    print(f" -> 목표치: {lod_targets}")

    print("\nStep 2: LOD 생성 시작 (QEM)...")
    
    # 경계 보호 가중치 (이전과 동일)
    boundary_weights = {
        "LOD0_High": 1.0, 
        "LOD1_Mid":  5.0,
        "LOD2_Low":  20.0
    }

    for level_name in LOD_ORDER: 
        target_count = lod_targets[level_name]
        weight = boundary_weights[level_name]
        
        # 원본 보존을 위해 복사본 사용
        mesh_copy = copy.deepcopy(mesh)

        # 원본보다 목표치가 작을 때만 줄임
        if original_count > target_count:
            mesh_lod = mesh_copy.simplify_quadric_decimation(
                target_number_of_triangles=target_count,
                boundary_weight=weight 
            )
        else:
            print(f" -> [{level_name}] 원본이 더 작아서 줄이지 않음.")
            mesh_lod = mesh_copy

        # 스타일링: LOD2는 각지게, 나머지는 부드럽게
        if level_name == "LOD2_Low":
            mesh_lod.vertex_normals = o3d.utility.Vector3dVector([]) # Flat Shading
        else:
            mesh_lod.compute_vertex_normals() # Smooth Shading

        # 저장
        final_count = len(mesh_lod.triangles)
        output_filename = f"{level_name}.obj"
        
        # 윈도우 경로 오류 방지 (절대경로 변환)
        save_path = os.path.abspath(os.path.join(output_folder, output_filename))
        
        o3d.io.write_triangle_mesh(save_path, mesh_lod, write_vertex_normals=True)
        print(f" -> [{level_name}] 완료 ({final_count:,} tris)")

    print(f"\n[완료] 결과물 폴더: {output_folder}")

# --- 실행부 ---
if __name__ == "__main__":
    INPUT_FILE = "instant_mesh_raw.obj"
    
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "input")
    input_path = os.path.join(input_folder, INPUT_FILE)
    output_dir = os.path.join(current_dir, "output_lods2")
    
    # 입력 확인 디버깅
    if os.path.exists(input_path):
        process_smart_lod(input_path, output_dir)
    else:
        print(f"[오류] 입력 파일을 찾을 수 없습니다.")
        print(f"경로: {input_path}")
        if os.path.exists(input_folder):
            print(f"폴더 내 파일 목록: {os.listdir(input_folder)}")
        else:
            print(f"'input' 폴더가 없습니다.")