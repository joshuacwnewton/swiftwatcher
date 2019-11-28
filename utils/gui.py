import cv2
from os import fspath


def select_corners(filepath):
    """OpenCV GUI function to select chimney corners from video frame."""

    def click_and_update(event, x, y, flags, param):
        """Callback function to record mouse coordinates on click, and to
        update instructions to user."""
        nonlocal corners

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 2:
                corners.append((int(x), int(y)))
                cv2.circle(image, corners[-1], 5, (0, 0, 255), -1)
                cv2.imshow("image", image)
                cv2.resizeWindow('image',
                                 int(0.5*image.shape[1]),
                                 int(0.5*image.shape[0]))

            if len(corners) == 1:
                cv2.setWindowTitle("image",
                                   "Click on corner 2")

            if len(corners) == 2:
                cv2.setWindowTitle("image",
                                   "Type 'y' to keep,"
                                   " or 'n' to select different corners.")

    stream = cv2.VideoCapture(fspath(filepath))
    success, image = stream.read()
    clone = image.copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_update)
    cv2.setWindowTitle("image", "Click on corner 1")

    corners = []

    while True:
        # Display image and wait for user input (click -> click_and_update())
        cv2.imshow("image", image)
        cv2.resizeWindow('image',
                         int(0.5 * image.shape[1]),
                         int(0.5 * image.shape[0]))
        cv2.waitKey(1)

        # Condition for when two corners have been selected
        if len(corners) == 2:
            key = cv2.waitKey(2000) & 0xFF

            if key == ord("n") or key == ord("N"):
                # Indicates selected corners are not good, so resets state
                image = clone.copy()
                corners = []
                cv2.setWindowTitle("image",
                                   "Click on corner 1")

            elif key == ord("y") or key == ord("Y"):
                # Indicates selected corners are acceptable
                break

        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) == 0:
            # Indicates window has been closed
            corners = []
            break

    cv2.destroyAllWindows()

    return corners
