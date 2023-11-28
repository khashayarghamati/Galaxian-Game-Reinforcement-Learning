

class Config:
    lr = 1e-4
    exploration_rate = 1
    exploration_rate_decay = 0.99999975
    exploration_rate_min = 0.1
    discount_factor = 0.99

    save_every = 5e5
    reply_buffer_size = 50000
    total_episode = 100000

    update_target_frequency = 5

    initial_epsilon = 0.1
    min_epsilon = 0.0001
    epsilon_discount_rate = 1e-7
    save_video_frequency = 500
    save_logs_frequency = 500
    show_loss_frequency = 10
    batch_size = 32
    initial_observe_episode = 100
    maximum_model = 5
    screen_width = 84
    screen_height = 84
