from visualization.linear_vis import main as linear_vis
from visualization.logistic_vis import main as logistic_vis

def main():
    print("Select Algorithm to Visualize:")
    print("1. Linear Regression")
    print("2. Logistic Regression")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        linear_vis()
    elif choice == "2":
        logistic_vis()
    # else:
        print("Invalid choice!")
if __name__ == "__main__":
    main()