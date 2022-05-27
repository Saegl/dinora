# Dinora

Dinora is alphazero-like chess engine. It uses 
keras/tensorflow for position evaluation and Monte Carlo Tree Search for 
calculating best move.

## Status
You can play against Dinora in standard chess variation, with or without increment.
I assume engine strength is about 1400 Lichess Elo, I evaluate engine rating 
basing on a few games against me, so it's not accurate.  
You can see example game below  
(10+0) Dinora (100-200 nodes in search) vs Me (2200 Rapid Lichess rating)  

![Chess game](https://github.com/Saegl/dinora/raw/main/assets/gif/gfychess-example.gif "Chess Game with Dinora engine")


### Features
- Working chess engine
- Minimal example of alpazero-like engine NN + MCTS
- All code included in this repo - for playing and training
- Everything written in python

## Installation

Currently tested only on Windows, but should work on Linux if you rewrite run_uci.bat.
You need docker installed on you machine, look at https://www.tensorflow.org/install/docker for more

1. Install preferred chess GUI with UCI support if you have one, 
I recommend Cute Chess - https://cutechess.com/, builded releases for Windows - 
https://github.com/cutechess/cutechess/releases

2. Clone this Repo
3. Open repo in terminal and run `docker-compose up -d`
4. To configure Dinora chess engine in CuteChess, browse
Tools > Settings > Engines > Add a new engine >  
Name: `Dinora Chess Engine` 
Command: `run_uci.bat`
Working Directory: `<folder where you clone the repo>`  
> Ok

Now you finally can start a game!
Game > new
Dinora can play for white and black pieces, only standard variant of chess  
Currently fixed time control for whole game supported, and time control with increment i.e
3+0, 5+0, 3+2, 5+5, 10+0 and others


# Acknowledgements

This engine based on https://github.com/Zeta36/chess-alpha-zero and 
https://github.com/dkappe/a0lite and Alphazero from Deepmind.

A lot of tutorials about chess engines from https://www.chessprogramming.org/ was super helpful.
