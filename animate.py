import cv2
import time
import numpy as np

def animate_frames(frames, actions_per_episode, break_duration=2):
    """
    Animates frames with a clear separation between episodes.

    Args:
        frames: List of frames to animate.
        actions_per_episode: List or array with the number of actions (frames) in each episode.
        break_duration: Time in seconds to wait between episodes, showing a break.
    """
    episode_number = 0
    current_action_count = 0  # Counter for actions in the current episode

    for frame in frames:
        current_action_count += 1

        # Convert from tensor to numpy array if necessary
        if not isinstance(frame, np.ndarray):
            frame = frame.numpy()

        frame = np.squeeze(frame)
        frame_resized = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)

        # Display the episode number on the frame
        cv2.putText(frame_resized, f"Episode: {episode_number + 1}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Episode", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' pressed
            break
        time.sleep(0.25)  # Speed of frames

        # Check if the episode should end
        if current_action_count == actions_per_episode[episode_number]:
            # Optionally, show a break or message between episodes
            cv2.destroyAllWindows()  # Close the window
            print(f"End of Episode {episode_number + 1}.")
            time.sleep(break_duration)  # Pause to indicate episode end
            
            # Reset action counter and move to the next episode
            current_action_count = 0
            episode_number += 1

            # Check if all episodes have been processed
            if episode_number >= len(actions_per_episode):
                break  # Exit the loop if all episodes are done
    
    cv2.destroyAllWindows()
