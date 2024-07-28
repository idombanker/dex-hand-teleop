import sys
from collections import deque
import time
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from threading import Thread
from hand_detector.hand_monitor import Record3DSingleHandMotionControl

app = Flask(__name__)
socketio = SocketIO(app)
plot_html = ""
trajectory = deque(maxlen=100)  # To store the trajectory of the first joint
runtimes = []  # To store the running time for each iteration


@app.route('/')
def index():
    global plot_html
    return render_template_string('''
    <html>
        <head>
            <title>Real-Time Hand Visualizer</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="//cdnjs.cloudflare.com/ajax/libs/socket.io/3.1.3/socket.io.min.js"></script>
        </head>
        <body>
            <div id="plotly-div" style="width:800px; height:800px;"></div>
            <script type="text/javascript">
                var socket = io.connect('http://' + document.domain + ':' + location.port);
                socket.on('update_plot', function(data) {
                    var plotly_data = JSON.parse(data.plot_html);
                    Plotly.react('plotly-div', plotly_data.data, plotly_data.layout);
                });
            </script>
        </body>
    </html>
    ''')

def run_flask():
    socketio.run(app, debug=False, use_reloader=False)

class RealTimeHandVisualizer:
    def __init__(self, hand_mode: str):
        self.motion_control = Record3DSingleHandMotionControl(hand_mode)
        self.setup_flask()
        
    def setup_flask(self):
        thread = Thread(target=run_flask)
        thread.daemon = True
        thread.start()

    def run(self):
        # global runtimes

        while True:
            # start_time = time.time()  # Record the start time
    
            success, output = self.motion_control.normal_step()
            if success:
                rgb = output["rgb"]
                joints = output["joint"]
                vertices = output.get("vertices", None)
                faces = output.get("faces", None)
                normals = output.get("normals", None)

                
                self.display_plot(vertices, faces, normals, joints)
            else:
                print("No hand detected.")

            self.display_image(rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # runtime = time.time() - start_time  # Calculate the runtime
            # runtimes.append(runtime)  # Append the runtime to the list
             


        
        cv2.destroyAllWindows()
        # self.plot_runtimes()  # Plot runtimes when exiting

    def display_image(self, rgb):
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Draw joints on the image
        # for joint in joints:
        #     # print(joint)
        #     cv2.circle(bgr, (int(joint[0]), int(joint[1])), 5, (0, 255, 0), -1)
        
        # Display the image in the OpenCV window
        cv2.imshow('Hand Detection', bgr)

    def display_plot(self, vertices, faces, normals,joints):
        global trajectory

        if vertices is not None and faces is not None:
            fig = go.Figure(data=[go.Mesh3d(
                x=vertices[:, 0], 
                y=vertices[:, 1], 
                z=vertices[:, 2], 
                i=faces[:, 0], 
                j=faces[:, 1], 
                k=faces[:, 2], 
                intensity=vertices[:, 2], 
                colorscale='Viridis', 
                opacity=0.50,
                colorbar=dict(tickformat=".2f"),
                cmin=-4,  # Set the minimum value for the color scale
                cmax=4    # Set the maximum value for the color scale
            )])
            if normals is not None:
                fig.add_trace(go.Cone(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    u=normals[:, 0],
                    v=normals[:, 1],
                    w=normals[:, 2],
                    colorscale='Blues',
                    sizemode="absolute",
                    sizeref=2,
                    colorbar=dict(tickformat=".4f"),
                    cmin=-4,  # Set the minimum value for the color scale
                    cmax=4    # Set the maximum value for the color scale
                ))

        if joints is not None:
            # Plot all joints in red
            fig.add_trace(go.Scatter3d(
                x=joints[:, 0],
                y=joints[:, 1],
                z=joints[:, 2],
                mode='markers',
                marker=dict(size=3, color='red')
            ))

            # Update the trajectory of the first joint
            # trajectory.append((joints[0, 0], joints[0, 1], joints[0, 2]))

            # Plot the first joint in green
            fig.add_trace(go.Scatter3d(
                x=[joints[0, 0]],
                y=[joints[0, 1]],
                z=[joints[0, 2]],
                mode='markers',
                marker=dict(size=5, color='green')
            ))

            # # Plot the trajectory of the first joint
            # trajectory_np = np.array(trajectory)
            # fig.add_trace(go.Scatter3d(
            #     x=trajectory_np[:, 0],
            #     y=trajectory_np[:, 1],
            #     z=trajectory_np[:, 2],
            #     mode='lines',
            #     line=dict(color='green', width=2)
            # ))

            # # Highlight the first joint in green
            # fig.add_trace(go.Scatter3d(
            #     x=[joints[0, 0]],
            #     y=[joints[0, 1]],
            #     z=[joints[0, 2]],
            #     mode='markers+text',
            #     marker=dict(size=5, color='green'),
            #     text=[f"({joints[0, 0]:.2f}, {joints[0, 1]:.2f}, {joints[0, 2]:.2f})"],
            #     textposition="bottom center"
            # ))
            # Set stable camera view, axis ranges, fixed figure size, and fixed precision
            camera = dict(
                eye=dict(x=0.8, y=0.5, z=0.5)
            )
            fig.update_layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis=dict(nticks=100, range=[-1.5, 0], autorange=False, tickformat=".4f"),
                    yaxis=dict(nticks=100, range=[-1, 1], autorange=False, tickformat=".4f"),
                    zaxis=dict(nticks=100, range=[-1, 1], autorange=False, tickformat=".4f"),
                    camera=camera
                ),
                width=800,
                height=800,
                margin=dict(l=0, r=0, b=0, t=0)  # Remove margins to keep the layout consistent
            )

            # Render the Plotly figure as JSON
            plot_json = pio.to_json(fig)
            socketio.emit('update_plot', {'plot_html': plot_json})
    def plot_runtimes(self):
        # Plot the runtimes after the program exits using Matplotlib
        plt.figure(figsize=(10, 5))
        plt.plot(runtimes, marker='o', linestyle='-')
        plt.title('Runtime per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True)
        plt.savefig('runtimes.png')  # Save the plot as an image

        # Display the saved image using OpenCV
        runtime_image = cv2.imread('runtimes.png')
        cv2.imshow('Runtimes', runtime_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_mode = "right_hand"  # Change to "left_hand" if needed
    visualizer = RealTimeHandVisualizer(hand_mode)
    visualizer.run()
