from absl import app
from absl import flags
from PIL import Image, ImageOps
import copy
import gym_env_processor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Breakout',
    'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')


def main(argv):
    # Create environment.
    def environment_builder():
        return gym_env_processor.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=1,
            frame_stack=FLAGS.environment_frame_stack,
            seed=1,
            max_noop_steps=10,
            terminal_on_life_loss=True,
        )

    env = environment_builder()

    obs = env.reset()

    frames = []

    for i in range(29):
        a_t = env.action_space.sample()

        obs, _, done, _ = env.step(a_t)

        numpy_array = env.render(mode='rgb_array')
        frame = Image.fromarray(numpy_array)

        frames.append(frame)

        if len(frames) > 18:
            base_name = f'breakout_frame_{i+1}'
            frame.save(f'./temp/{base_name}.png')

            overlayed = stack_frames_example(copy.deepcopy(frames))
            overlayed.save(f'./temp/{base_name}_stacked.png')

            processed_frames = copy.deepcopy(frames)
            grid = action_repeat_example(processed_frames, 4, 4, 6)
            grid.save(f'./temp/{base_name}_grid.png')

            skiped_overlayed = stack_frames_example(processed_frames[-16::4])
            skiped_overlayed.save(f'./temp/{base_name}_action_repeat_and_stacked.png')

        if done:
            break


def remove_black_background(img):
    """Using pillow to remove black background."""
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)

    img.putdata(newData)

    return img


def add_black_background(img):
    """Using pillow to add black background."""
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 255))
        else:
            newData.append(item)

    img.putdata(newData)

    return img


def stack_frames_example(frames):
    overlayed = None
    for i in range(len(frames)):
        if overlayed is None:
            overlayed = frames[i]
        else:
            frame = frames[i].convert('RGBA')  # need for paste operation.
            frame = remove_black_background(frame)
            x, y = frame.size
            overlayed.paste(frame, (0, 0, x, y), mask=frame)

    overlayed = add_black_background(overlayed)

    return overlayed


def image_grid(imgs, rows, cols):
    # assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGBA', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def action_repeat_example(frames, num_repeat, grid_row, grid_col):
    processed_frames = []
    for i, frame in enumerate(frames):
        if i > 0 and i % num_repeat != 0:
            frame.putalpha(75)  # make opacity 0.3

        # add border to the frames.
        frame = ImageOps.expand(frame, border=5, fill='white')
        processed_frames.append(frame)

    return image_grid(processed_frames, grid_row, grid_col)


if __name__ == '__main__':
    app.run(main)
