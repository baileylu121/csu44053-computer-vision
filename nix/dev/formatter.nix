{ inputs, ... }:
{
  imports = [ inputs.treefmt-nix.flakeModule ];

  perSystem.treefmt = {
    projectRootFile = "flake.nix";

    settings.global.excludes = [
      "*.envrc"
      "*.envrc."
    ];

    programs = {
      clang-format.enable = true;
      deadnix.enable = true;
      meson.enable = true;
      nixfmt.enable = true;
      statix.enable = true;
    };
  };
}
