{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, devenv, systems, ... } @ inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem
        (system:
          let
            pkgs = nixpkgs.legacyPackages.${system};
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                ({ pkgs, ... }: {
                  # https://devenv.sh/reference/options/
                  packages = [
                    pkgs.pyright
                    pkgs.gcc-unwrapped.lib
                    pkgs.zlib
                  ];

                  pre-commit.hooks = {
                    nixpkgs-fmt.enable = true;
                    # lint shell scripts
                    shellcheck.enable = true;
                    # format Python code
                    black.enable = true;
                  };

                  languages.python.enable = true;
                  languages.python.venv.enable = true;
                  languages.python.venv.requirements = ''
                    scipy
                    numpy
                    matplotlib
                  '';
                })
              ];
            };
          });
    };
}
