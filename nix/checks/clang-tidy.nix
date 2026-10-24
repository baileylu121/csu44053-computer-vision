{
  perSystem =
    { pkgs, self', ... }:
    {
      checks.clang-tidy = self'.packages.pool-balls.overrideAttrs (prev: {
        pname = "pool-balls-clang-tidy";

        nativeBuildInputs = prev.nativeBuildInputs ++ [ pkgs.clang-tools ];

        buildPhase = ''
          runHook preBuild
          ninja clang-tidy
          runHook postBuild
        '';

        installPhase = ''
          touch "$out"
        '';
      });
    };
}
