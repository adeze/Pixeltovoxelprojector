import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxel grid and mesh extraction")
    parser.add_argument("metadata_path", help="Path to metadata JSON file")
    parser.add_argument("images_folder", help="Folder containing images")
    parser.add_argument("output_bin", help="Output voxel grid .bin file")
    parser.add_argument(
        "--use-mcubes", action="store_true", help="Extract mesh with PyMCubes"
    )
    parser.add_argument(
        "--output-mesh",
        default="output_mesh.obj",
        help="Output mesh OBJ file (if --use-mcubes is set)",
    )
    args = parser.parse_args()

    process_all(
        args.metadata_path,
        args.images_folder,
        args.output_bin,
        use_mcubes=args.use_mcubes,
        output_mesh=args.output_mesh,
    )
