import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_parabola(points, epsilon=1e-5):
    # Fit a conic to the extracted points
    A = np.vstack(
        [points[:, 0] ** 2, points[:, 0] * points[:, 1], points[:, 1] ** 2,
         points[:, 0], points[:, 1], np.ones(points.shape[0])]).T
    _, _, V = np.linalg.svd(A)
    coefficients = V[-1, :]
    A, B, C, D, E, F = coefficients

    # Compute the discriminant
    discriminant = B ** 2 - 4 * A * C
    print("The discriminant is:")
    print(discriminant)

    # Check if the discriminant is close to zero within epsilon value,
    # indicating a parabola
    if np.isclose(discriminant, 0, rtol=0, atol=epsilon):
        return True
    else:
        return False


def find_camera_location(points_3d, points_2d):
    """
    Find the camera's location (extrinsic parameters) using DLT algorithm.

    Args:
        points_3d (np.ndarray): Array of 3D points in homogeneous coordinates, shape (n, 4)
        points_2d (np.ndarray): Array of 2D points in homogeneous coordinates, shape (n, 3)

    Returns:
        np.ndarray: Array representing the camera's location (translation vector) and orientation (rotation matrix),
        shape (3, 4)
    """
    assert points_3d.shape[0] == points_2d.shape[
        0], "Number of points must be the same in 3D and 2D space"

    # Normalize the 3D points to have zero mean and unit standard deviation
    points_3d_normalized = np.array(points_3d[:, :3], dtype=np.float64)
    mean_3d = np.mean(points_3d_normalized, axis=0)
    std_3d = np.std(points_3d_normalized, axis=0)
    T_3d = np.eye(4)
    T_3d[:3, 3] = -mean_3d
    T_3d[:3, :3] = np.diag(1 / std_3d)
    points_3d_normalized[:, :3] = points_3d_normalized[:, :3] - mean_3d
    points_3d_normalized[:, :3] = points_3d_normalized[:, :3] @ np.diag(1 / std_3d)


    points_2d_normalized = np.array(points_2d[:, :2], dtype=np.float64)
    mean_2d = np.mean(points_2d_normalized, axis=0)
    std_2d = np.std(points_2d_normalized, axis=0)
    T_2d = np.eye(4)
    T_2d[:2, 2] = -mean_2d
    T_2d[:2, :2] = np.diag(1 / std_2d)
    points_2d_normalized[:, :2] = points_2d_normalized[:, :2] - mean_2d
    points_2d_normalized[:, :2] = points_2d_normalized[:, :2] @ np.diag(1 / std_2d)
    points_2d_normalized = np.hstack((points_2d_normalized, np.ones((6, 1))))

    # Construct the A matrix
    num_points = points_3d.shape[0]
    A = np.zeros((2 * num_points, 12))
    for i in range(num_points):
        X, Y, Z,_ = points_3d[i]
        u, v, w = points_2d[i]
        A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, u * X, u * Y, u * Z, u]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v]

    # Solve the linear system using SVD
    u, s, vh = np.linalg.svd(A)
    P = vh[-1].reshape((3, 4))

    intrinsic_matrix = P[:, 0:3]
    extrinsic_matrix = P[:, 3]

    # Extract the camera location in homogeneous coordinates
    camera_location_homogeneous = -np.linalg.inv(
        intrinsic_matrix) @ extrinsic_matrix

    # Convert camera location to projective space by adding homogeneous
    # coordinate of 1
    camera_location_projective = np.concatenate(
        (camera_location_homogeneous, np.array([1])), axis=0)

    return camera_location_projective


def direct_linear_calibration(image_points, world_points):
    """
    Direct Linear Calibration (DLC) to estimate camera parameters.

    Args:
        image_points (np.array): Nx3 array of 2D image points.
        world_points (np.array): Nx4 array of 3D world points.

    Returns:
        np.array: 3x3 intrinsic camera matrix.
        np.array: 3x4 extrinsic camera matrix.
        np.array: 4x1 camera location in projective space.
    """
    # Convert image points and world points to homogeneous coordinates
    image_points_homogeneous = np.concatenate((image_points, np.ones((image_points.shape[0], 1))), axis=1)
    world_points_homogeneous = np.concatenate((world_points, np.ones((world_points.shape[0], 1))), axis=1)

    # Build the A matrix
    A = np.zeros((2 * image_points.shape[0], 12))
    for i in range(image_points.shape[0]):
        A[2*i, 0:4] = -world_points_homogeneous[i, :-1]
        A[2*i, 8:12] = image_points_homogeneous[i, 0] * world_points_homogeneous[i, :]
        A[2*i + 1, 4:8] = -world_points_homogeneous[i, :-1]
        A[2*i + 1, 8:12] = image_points_homogeneous[i, 1] * world_points_homogeneous[i, :]

    # Perform SVD on A matrix
    _, _, V = np.linalg.svd(A)

    # Extract the camera parameters
    camera_matrix = np.reshape(V[-1, :], (3, 4))
    intrinsic_matrix = camera_matrix[:, 0:3]
    extrinsic_matrix = camera_matrix[:, 3]

    # Extract the camera location in homogeneous coordinates
    camera_location_homogeneous = -np.linalg.inv(intrinsic_matrix) @ extrinsic_matrix

    # Convert camera location to projective space by adding homogeneous
    # coordinate of 1
    camera_location_projective = np.concatenate((camera_location_homogeneous, np.array([1])), axis=0)

    return intrinsic_matrix, extrinsic_matrix, camera_location_projective


if __name__ == '__main__':
    points = np.array([[49.3268, 199.975], [332.713, 563.754], [761.811,
                                                                698.413],
              [1279.34, 550.69], [1537.6, 205.0]])

    # Usage example
    image_path = 'parabola_image.jpg'
    result = is_parabola(points)
    if result:
        print("The imaged curve is a parabola.")
    else:
        print("The imaged curve is not a parabola.")

    # Define the points in 3D homogeneous space
    points_3d = np.array([
        [0, 105, 10, 1],
        [0, 105, 80, 1],
        [70, 105, 80, 1],
        [0, 130, 80, 1],
        [43, 115, 80, 1],
        [60, 115, 106, 1]
    ])

    # Define the points in 2D image space
    points_2d = np.array([
        [540, 765, 1],
        [547, 460, 1],
        [88, 465, 1],
        [545, 386, 1],
        [270, 429, 1],
        [123, 266, 1]
    ])
#
    # Find the camera's location (extrinsic parameters)
    camera_location = find_camera_location(points_3d, points_2d)
    # Print the camera's location
    print("Camera's Location (Extrinsic Parameters):")
    print(camera_location)
