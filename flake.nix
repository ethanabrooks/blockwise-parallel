{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    poetry2nix,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          permittedInsecurePackages = [
            "openssl-1.1.1u"
          ];
        };
      };
      inherit (poetry2nix.lib.mkPoetry2Nix {inherit pkgs;}) mkPoetryEnv;
      python = pkgs.python311;
      poetryEnv = mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
      };

      buildInputs = with pkgs; [
        alejandra
        nodejs_20
        ngrok
        yarn
        libuuid
        jq
        patchelf
        zip
        sqlcmd
        alejandra
        poetry
        poetryEnv
      ];
    in rec {
      devShell = pkgs.mkShell rec {
        inherit buildInputs;
        PYTHONBREAKPOINT = "ipdb.set_trace";
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
