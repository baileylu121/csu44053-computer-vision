{
  perSystem =
    {
      pkgs,
      self',
      config,
      ...
    }:
    let
      inherit (self'.packages) pool-balls;
    in
    {
      devShells.default = (pkgs.mkShell.override { inherit (pool-balls) stdenv; }) {
        inputsFrom = [ pool-balls ];

        packages = with pkgs; [
          ccls
          clang-tools
          config.treefmt.build.wrapper
        ];

        hardeningDisable = [ "fortify" ];

        shellHook = ''
          ln -sfn build/compile_commands.json compile_commands.json
        '';
      };
    };
}
