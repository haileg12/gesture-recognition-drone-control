from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
import threading

# Global variables for movement
move_forward = move_backward = move_right = move_left = False


def initialize_drone_control(sim):
    sim.setStepping(True)
    sim.startSimulation()

    target_handle = sim.getObject('/target')
    base_handle = sim.getObject('/base')

    return target_handle, base_handle


def set_target_position(sim, target_handle, x, y, z):
    sim.setObjectPosition(target_handle, -1, [x, y, z])


def on_press(key):
    global move_forward, move_backward, move_right, move_left
    try:
        if key.char == 'w':
            move_forward = True
        elif key.char == 's':
            move_backward = True
        elif key.char == 'd':
            move_right = True
        elif key.char == 'a':
            move_left = True
    except AttributeError:
        pass


def on_release(key):
    global move_forward, move_backward, move_right, move_left
    try:
        if key.char == 'w':
            move_forward = False
        elif key.char == 's':
            move_backward = False
        elif key.char == 'd':
            move_right = False
        elif key.char == 'a':
            move_left = False
        if key == keyboard.Key.esc:
            # Stop listener
            return False
    except AttributeError:
        pass


def keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    target_handle, base_handle = initialize_drone_control(sim)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    speed = 0.1

    try:
        while True:
            sim.step()
            target_pos = sim.getObjectPosition(target_handle, -1)
            delta = speed * sim.getSimulationTimeStep()
            if move_forward:
                target_pos[1] += delta
            if move_backward:
                target_pos[1] -= delta
            if move_right:
                target_pos[0] += delta
            if move_left:
                target_pos[0] -= delta
            set_target_position(sim, target_handle, target_pos[0], target_pos[1], target_pos[2])

    except KeyboardInterrupt:
        pass
    finally:
        sim.stopSimulation()


if __name__ == "__main__":
    main()
