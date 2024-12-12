from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
from math import cos, sin, pi
import numpy as np
import math

transformationMatrix = [[1,0,0],[0,1,0],[0,0,1]]
# transformationMatrix = [[1,0,5],[0,1,5],[0,0,1]]

# Drawing based on user choice
def draw(X1,Y1, X2, Y2):
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 0.0, 0.0)  # Red for all shapes
    
    glLineWidth(7.0)


    glBegin(GL_LINES)
    glVertex2f(X1, Y1)
    glVertex2f(X2, Y2)
    glEnd()

    P1 = [X1,Y1,1]
    
    P1_prime = np.dot(transformationMatrix,P1)

    x1 = P1_prime[0]
    y1 = P1_prime[1]

    P2 = [X2,Y2,1]

    P2_prime = np.dot(transformationMatrix,P2)

    x2 = P2_prime[0]
    y2 = P2_prime[1]

    glColor3f(0.0, 0.0, 1.0)  # Red for all shapes
    
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()
    
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

    # Get user choice
    print("Choose among following geometric transformation")
    print("1. Translation")
    print("2. Rotation")
    print("3. Scaling")
    choice = int(input("Enter your choice (1/2/3): "))

    if(choice==1):
        tx = int(input("Enter your translation along x(tx): "))
        ty = int(input("Enter your translation along y(ty): "))
        global transformationMatrix
        transformationMatrix = [[1,0,tx],[0,1,ty],[0,0,1]]

    if(choice==2):
        theta = int(input("Enter your rotation angle(+ve for anticlockwise -ve for clockwise): "))
        ntheta = 0 - theta
        transformationMatrix = [[math.cos(theta),math.sin(ntheta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]]

    if(choice==3):
        sx = int(input("Enter your scaling along x(sx): "))
        sy = int(input("Enter your scaling along y(sy): "))
        transformationMatrix = [[sx,0,0],[0,sy,0],[0,0,1]]

    X1 = int(input("Enter x co-ordinate for the 1st point of line:"))
    Y1 = int(input("Enter Y co-ordinate for the 1st point of line:"))

    X2 = int(input("Enter x co-ordinate for the 2nd point of line:"))
    Y2 = int(input("Enter Y co-ordinate for the 2nd point of line:"))

    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        draw(X1,Y1,X2,Y2)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
