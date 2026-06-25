import open3d as o3d
try:
    office = o3d.data.OfficePointClouds()
    print("Office paths:", len(office.paths))
    print(office.paths[0])
except Exception as e:
    print(e)

try:
    ply = o3d.data.PLYPointCloud()
    print("PLY path:", ply.path)
except Exception as e:
    print(e)
