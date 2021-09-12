from dinora.utils import disable_tensorflow_log

disable_tensorflow_log()
from tensorflow import keras

from dinora.net import ChessModel
from dinora.dataset import create_dataset_from_games
from dinora.model import build_model, LightConfig
from dinora.selfplay import gen_game

nodes = 5
net = ChessModel("models/best_light_model.h5")
net.model = build_model(LightConfig)  # Start from zero
net.model.compile(
    keras.optimizers.Adam(0.05),
    loss=["categorical_crossentropy", "mean_squared_error"],
    metrics=["accuracy"],
)

c_puct = 2.0
softmax_temp = 1.6
dirichlet_alpha = 0.3
noise_eps = 0.2

stats = {}
i = 0
while True:
    selfplay_games = [
        gen_game(nodes, net, c_puct, dirichlet_alpha, noise_eps, softmax_temp)
        for _ in range(3)
    ]

    dataset = create_dataset_from_games(selfplay_games)
    dataset = dataset.batch(4096)
    net.model.fit(dataset, epochs=1)

    for g in selfplay_games:
        res = g.headers["Result"]
        stats[res] = stats.setdefault(res, 0) + 1

    if i % 256 == 0:
        net.model.save_weights("weights.h5")
        print(stats)
        print(selfplay_games[0])
    i += 1
