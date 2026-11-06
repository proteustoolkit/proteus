
// Points
Point(1) = {0,0,0};
Point(2) = {0,1,0};
Point(3) = {1,1,0};
Point(4) = {1,0,0};
Point(5) = {0,0,1};
Point(6) = {0,1,1};
Point(7) = {1,1,1};
Point(8) = {1,0,1};

// Lines

// Surfaces
Line(1) = {4,1};
Line(2) = {1,2};
Line(3) = {2,3};
Line(4) = {3,4};
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line(6) = {6,2};
Line(7) = {2,3};
Line(8) = {3,7};
Line(9) = {7,6};
Line Loop(2) = {6, 7, 8, 9};
Plane Surface(2) = {2};
Line(11) = {7,3};
Line(12) = {3,4};
Line(13) = {4,8};
Line(14) = {8,7};
Line Loop(3) = {11, 12, 13, 14};
Plane Surface(3) = {3};
Line(16) = {8,4};
Line(17) = {4,1};
Line(18) = {1,5};
Line(19) = {5,8};
Line Loop(4) = {16, 17, 18, 19};
Plane Surface(4) = {4};
Line(21) = {5,1};
Line(22) = {1,2};
Line(23) = {2,6};
Line(24) = {6,5};
Line Loop(5) = {21, 22, 23, 24};
Plane Surface(5) = {5};
Line(26) = {8,5};
Line(27) = {5,6};
Line(28) = {6,7};
Line(29) = {7,8};
Line Loop(6) = {26, 27, 28, 29};
Plane Surface(6) = {6};

// Volumes
Surface Loop(7) = {1, 2, 3, 4, 5, 6};
Volume(1) = {7};

// Physical Groups
Physical Point(1) = {1, 2, 3, 4};
Physical Point(6) = {5, 6, 7, 8};
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};
Physical Surface(5) = {5};
Physical Surface(6) = {6};
Physical Volume(1) = {1};

// Other Options
Mesh.CharacteristicLengthMax = 0.5;
Coherence;
