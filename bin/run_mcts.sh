#!/usr/bin/env bash

if [[ -f /etc/NIXOS && -d /run/opengl-driver/lib && -n "$NIX_LD_LIBRARY_PATH" ]]; then
    # Needed for NixOS with nixld to work with Nvidia GPU in pytorch
    # Most distros don't need this line
    # And even NixOS users don't often use NIX_LD_LIBRARY_PATH
    # This is personal fix
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:$NIX_LD_LIBRARY_PATH
fi

.venv/bin/python -m dinora --searcher mcts --model torch
