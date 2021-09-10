from dinora.utils import disable_tensorflow_log
disable_tensorflow_log()
from tensorflow import keras

from dinora.net import ChessModel
from dinora.dataset import create_dataset_from_games
from dinora.model import build_model, LightConfig
from dinora.selfplay import gen_game

nodes = 5
net = ChessModel('models/best_light_model.h5')
net.model = build_model(LightConfig)  # Start from zero
net.model.compile(
    keras.optimizers.Adam(0.05),
    loss=["categorical_crossentropy", "mean_squared_error"],
    metrics=["accuracy"],
)
c = 2.0

stats = {}
i = 0
while True:
    game = gen_game(nodes, net, c) # Selfplay game 
    games = [game]
    dataset = create_dataset_from_games(games)
    dataset = dataset.batch(4096)
    net.model.fit(dataset, epochs=1)
    
    res = game.headers['Result']
    stats[res] = stats.setdefault(res, 0) + 1
    
    if i % 256 == 0:
        net.model.save_weights('weights.h5')
        print(stats)
    i += 1
