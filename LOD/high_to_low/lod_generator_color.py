import open3d as o3d
import os
import time
import copy

# LOD 단계 정의
LOD_ORDER = ["LOD0_High", "LOD1_Mid", "LOD2_Low"]

def calculate_target_counts(original_count):
    targets = {}
    targets["LOD0_High"] = min(original_count, 50000)
    mid_target = int(original_count * 0.2)
    targets["LOD1_Mid"] = min(mid_target, 10000)
    low_target = int(original_count * 0.02)
    low_target = max(low_target, 500) 
    targets["LOD2_Low"] = min(low_target, 2000) 
    return targets

def simple_cleanup(mesh):
    """기본 메쉬 정리 (중복 제거)"""
    print(" -> 기본 메쉬 정리 중...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh

def process_smart_lod_color(input_path, output_folder):
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    print(f"\nStep 1: 모델 로드 중... [{os.path.basename(input_path)}]")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if not mesh.has_triangles():
        print("[오류] 메쉬 정보를 읽을 수 없습니다.")
        return
    
    # 색상 정보 확인
    if mesh.has_vertex_colors():
        print(" -> [Info] Vertex Color 감지됨. 색상을 보존합니다.")
    else:
        print(" -> [Info] 색상 정보가 없습니다.")

    # 정리 및 법선 계산
    mesh = simple_cleanup(mesh)
    mesh.compute_vertex_normals() 

    original_count = len(mesh.triangles)
    print(f" -> 원본 폴리곤 개수: {original_count:,}개")

    lod_targets = calculate_target_counts(original_count)
    print(f" -> 목표치: {lod_targets}")

    print("\nStep 2: LOD 생성 시작 (Color Preserved)...")
    
    boundary_weights = {"LOD0_High": 1.0, "LOD1_Mid": 5.0, "LOD2_Low": 20.0}

    for level_name in LOD_ORDER: 
        target_count = lod_targets[level_name]
        weight = boundary_weights[level_name]
        
        mesh_copy = copy.deepcopy(mesh)

        if original_count > target_count:
            mesh_lod = mesh_copy.simplify_quadric_decimation(
                target_number_of_triangles=target_count,
                boundary_weight=weight 
            )
        else:
            mesh_lod = mesh_copy

        if level_name == "LOD2_Low":
            mesh_lod.vertex_normals = o3d.utility.Vector3dVector([])
        else:
            mesh_lod.compute_vertex_normals()

        # 저장 (색상 포함)
        output_filename = f"{level_name}.obj"
        save_path = os.path.abspath(os.path.join(output_folder, output_filename))
        
        # write_vertex_colors=True가 핵심
        o3d.io.write_triangle_mesh(save_path, mesh_lod, write_vertex_normals=True, write_vertex_colors=True)
        print(f" -> [{level_name}] 완료 ({len(mesh_lod.triangles):,} tris)")

    print(f"\n[완료] 결과물 폴더: {output_folder}")

if __name__ == "__main__":
    INPUT_FILE = "instant_mesh_raw.obj"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "input", INPUT_FILE)
    output_dir = os.path.join(current_dir, "output_lods_color")
    
    if os.path.exists(input_path):
        process_smart_lod_color(input_path, output_dir)
    else:
        print(f"[오류] 파일 없음: {input_path}")