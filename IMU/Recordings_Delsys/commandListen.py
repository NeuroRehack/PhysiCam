import socket
import threading
import time
import cv2
from datetime import datetime   


global is_recording
is_recording = False


def start_recording():
    global is_recording
    currently_recording = False

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Open the webcam
    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"resolution: {int(width)} x {int(height)}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        # out.write(frame)  # Write the captured frame to file

        # Display the resulting frame
        #cv2.imshow('frame', frame)

        if not currently_recording and is_recording:
            print("Starting recording")

            now = datetime.now()
            print(now)

            now = now.strftime("%y%m%d_%H-%M-%S")
            out = cv2.VideoWriter(
                f'videos/{now}.avi', 
                fourcc, 
                30.0,               # frame rate (30 fps)
                (width, height),    # frame size (width, height)
            )
            
            currently_recording = True
            start_time = time.time()

            # List to store all timestamps
            timestamps = list()

            # Create file to store timestamps
            #f = open(f"time_stamps/{now}.txt", "a")

        elif is_recording and currently_recording:
            out.write(frame)

            # Append the elapsed time to the timestamps list
            timestamps.append(time.time() - start_time)

            # Write the elapsed time to the timestamps file
            #f.write(f"{str(time.time() - start_time)}\n")

        elif currently_recording and not is_recording:
            print("Stopping recording")

            out.release()
            currently_recording = False

            # Write the timestamps to a text file at the end of the recording
            with open(f"time_stamps/{now}.txt", "a") as f:
                for t in timestamps:
                    f.write(f"{t}\n")

            # Close the file at the end of the recording
            #f.close()

        else:
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording
                break

    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def receive_data(server_socket):
    global is_recording
    while True:
        try:
            data, address = server_socket.recvfrom(1024)  # Adjust buffer size if needed
            now = datetime.now()
            print(f"Received message @ {now}: {data.decode('utf-8')} from {address[0]}")
            if "StartRecording" in data.decode('utf-8'):
                is_recording = True
            elif "StopRecording" in data.decode('utf-8'):
                is_recording = False
        except socket.timeout:
            continue

def command_rcv_loop():
    UDP_IP = "0.0.0.0"  # Listen on all available interfaces
    UDP_PORT = 1510  # Must match the port used in the Raspberry Pi script

    # Create a UDP socket and bind it to the IP address and port
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_IP, UDP_PORT))

    # Set a timeout of 5 seconds for the receive operation
    server_socket.settimeout(5)

    # Create a separate thread for receiving data
    receive_thread = threading.Thread(target=receive_data, args=(server_socket,))
    receive_thread.start()

    # Wait for the Ctrl+C signal in the main thread
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Exiting...")
    finally:
        # Cleanup and join the receive thread
        server_socket.close()
        receive_thread.join()

if __name__ == "__main__":
    video_thread = threading.Thread(target=start_recording)
    video_thread.start()
    command_rcv_loop()
    video_thread.join()
