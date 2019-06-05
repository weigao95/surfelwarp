## SurfelWarp
An implementation of our SurfelWarp paper. This version aims at state-of-art level online dynamic reconstruction.


### Main new features compared with previous version:
1. The non-rigid solver uses rgb (or intense) features and sparse point correspondence.
2. The image processor and non-rigid solver are thread-level parallel. For better load balance, more functionalities shall be placed in image processor.
3. The geometry model automatically reset itself to meet the current depth observations.
4. The usage of stream for performance boosting (next iteration).

### Design and Code Style
1. The C++ code should be Object-Oriented, while the cuda code should be procedural. All cuda functions shall have the API for stream inputs.
3. Every non-trivial cuda function shall have a C++ implementation for testing. They shall share the same header, but implemented in different *.cpp and *.cu files.
4. In the first iterations, only implement the functionalities. Provide the API for stream but not actually use them.
