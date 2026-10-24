{
  perSystem =
    { pkgs, ... }:
    let
      inherit (pkgs) lib;
    in
    {
      packages = rec {
        pool-balls = (pkgs.useWildLinker pkgs.stdenv).mkDerivation {
          pname = "pool-balls";
          version = "0.1.0";

          src = lib.fileset.toSource {
            root = ../.;
            fileset = lib.fileset.unions [
              ../meson.build
              ../.clang-tidy
              (lib.fileset.fileFilter (file: file.hasExt "cpp") ../.)
            ];
          };

          nativeBuildInputs = with pkgs; [
            meson
            ninja
            pkg-config
          ];

          buildInputs = [
            (pkgs.opencv.override { enableGtk3 = true; })
          ];

          meta.mainProgram = "app";
        };
        default = pool-balls;
      };
    };
}
