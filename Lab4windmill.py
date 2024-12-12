
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
from math import cos, sin, pi
import numpy as np
import math
import time


theta = 0.78
transformationMatrix = [[math.cos(theta),math.sin(-theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]]


X1 = 0
Y1 = 0
X2 = 30
Y2 = 5
X3 = 30 
Y3 = -5


# Drawing based on user choice
def draw():
    global X1,X2, Y1, Y2 , X3, Y3

    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT)

    #windmill base
    glColor3f(1.0, 1.0, 0.0)  # yellow for base
    glBegin(GL_POLYGON)
    glVertex2f(-5, -90) #for drawing rectangular base part
    glVertex2f(5, -90)
    glVertex2f(5, 0)
    glVertex2f(-5, 0)
 
    glEnd()


    glColor3f(1.0, 0.0, 0.0)  # Red for all shapes
    
    glBegin(GL_POLYGON)
    glVertex2f(X1,Y1) #for drawing rectangular base part
    glVertex2f(X2,Y2)
    glVertex2f(X3,Y3)
    glEnd()


    glBegin(GL_POLYGON)
    glVertex2f(Y1,X1) #for drawing rectangular base part
    glVertex2f(Y2,X2)
    glVertex2f(Y3,X3)
    glEnd()


    glBegin(GL_POLYGON)
    glVertex2f(-X1,Y1) #for drawing rectangular base part
    glVertex2f(-X2,Y2)
    glVertex2f(-X3,Y3)
    glEnd()


    glBegin(GL_POLYGON)
    glVertex2f(Y1,-X1) #for drawing rectangular base part
    glVertex2f(Y2,-X2)
    glVertex2f(Y3,-X3)
    glEnd()




    P1 = [X1,Y1,1]
    P1_prime = np.dot(transformationMatrix,P1)
    X1 = P1_prime[0]
    Y1 = P1_prime[1]

    P2 = [X2,Y2,1]
    P2_prime = np.dot(transformationMatrix,P2)
    X2 = P2_prime[0]
    Y2 = P2_prime[1]

    P3 = [X3,Y3,1]
    P3_prime = np.dot(transformationMatrix,P3)
    X3 = P3_prime[0]
    Y3 = P3_prime[1]

    
    
    glFlush()

def main():
    if not glfw.init():
        return

    window = glfw.create_window(600, 600, "Drawing Algorithms", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    gluOrtho2D(-100, 100, -100, 100)

    #frame interval (default normal speed)
    frame_interval = 0.04

     # Get user choice
    print("Choose two different speed")
    print("1. Normal")
    print("2. High")
    choice = int(input("Enter your choice (1/2): "))

    if(choice==2):
        frame_interval = 0.02


    # Initialize timer
    last_time = time.time()

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # draw()
        # glfw.swap_buffers(window)

        current_time = time.time()
        if current_time - last_time >= frame_interval:
            draw()  # Update and render animation
            glfw.swap_buffers(window)
            last_time = current_time  # Reset the timer

    glfw.terminate()

if __name__ == "__main__":
    main()
