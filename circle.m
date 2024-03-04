% Load the image
img = imread('Image1.jpg');

% Display the image
figure;
imshow(img);
hold on

% Select four points on the ground plane
disp('Select four points on the ground plane, starting from point A:');
points = ginput(6);
scatter(points(:,1), points(:,2),'r','filled');