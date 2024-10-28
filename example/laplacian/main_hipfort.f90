program main
    use hipfort
    use hipfort_check
    implicit none
    integer, parameter :: nx=512, ny=512, nz=512
    real, parameter :: tol=3.e-1
    ! https://web.media.mit.edu/~crtaylor/calculator.html
    ! locations of sampled points: -4,-3,-2,-1,0,1,2,3,4
    ! derivative order: 2
    ! fractions are break down into its simplest form
    real, parameter :: c0 = -205./72., c1 =  8./5.,  c2 = -1./5.
    real, parameter :: c3 =  8./315.,  c4 = -1./560.
    real, allocatable :: hcoeffs_x(:), hcoeffs_y(:), hcoeffs_z(:), out(:)
    real, pointer, dimension(:) :: du, df, dcoeffs_x, dcoeffs_y, dcoeffs_z
    real :: hx, hy, hz, expected_f
    integer :: i, j, k, count, pos
    type(dim3) :: grid  = dim3(320,1,1) 
    type(dim3) :: block = dim3(4,8,8) 

    interface
    ! dim3(320), dim3(256), 0, 0
    subroutine laplacian(grid,block,shmem,stream, &
                         f, u, coeffs_x, coeffs_y, coeffs_z, &
                         nx, ny, nz) bind(c)
      use iso_c_binding
      use hipfort_types
      implicit none
      type(c_ptr),value :: f, u, coeffs_x, coeffs_y, coeffs_z
      integer(c_int), value :: nx, ny, nz, shmem
      type(dim3) :: grid, block
      type(c_ptr),value :: stream
    end subroutine

    subroutine test_function(grid,block,shmem,stream, & 
                             u, nx, ny, nz, hx, hy, hz) bind(c)
      use iso_c_binding
      use hipfort_types
      implicit none
      type(c_ptr),value :: u
      real(c_float), value :: hx, hy, hz
      integer(c_int), value :: nx, ny, nz, shmem
      type(dim3) :: grid, block
      type(c_ptr),value :: stream
    end subroutine
    end interface

    nullify(du)
    nullify(df)
    hx = 1.0 / (nx - 1);
    hy = 1.0 / (ny - 1);
    hz = 1.0 / (nz - 1);

    allocate(hcoeffs_x(1:5), hcoeffs_y(1:5), hcoeffs_z(1:5), out(nx*ny*nz))

    hcoeffs_z(1) = c0/hz/hz; hcoeffs_x(1) = c0/hx/hx; hcoeffs_y(1) = c0/hy/hy
    hcoeffs_z(2) = c1/hz/hz; hcoeffs_x(2) = c1/hx/hx; hcoeffs_y(2) = c1/hy/hy
    hcoeffs_z(3) = c2/hz/hz; hcoeffs_x(3) = c2/hx/hx; hcoeffs_y(3) = c2/hy/hy
    hcoeffs_z(4) = c3/hz/hz; hcoeffs_x(4) = c3/hx/hx; hcoeffs_y(4) = c3/hy/hy
    hcoeffs_z(5) = c4/hz/hz; hcoeffs_x(5) = c4/hx/hx; hcoeffs_y(5) = c4/hy/hy

  ! Allocate array space on the device
    call hipCheck(hipMalloc(du,nz*nx*ny))
    call hipCheck(hipMalloc(df,nz*nx*ny))
    call hipCheck(hipMalloc(dcoeffs_x, 5))
    call hipCheck(hipMalloc(dcoeffs_y, 5))
    call hipCheck(hipMalloc(dcoeffs_z, 5))
  
    ! Transfer data from host to device memory
    call hipCheck(hipMemcpy(dcoeffs_x, hcoeffs_x, 5, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dcoeffs_y, hcoeffs_y, 5, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dcoeffs_z, hcoeffs_z, 5, hipMemcpyHostToDevice))

    ! Cartesian Coordinate System (3D):
    !          y
    !         /
    !        /
    !       +-------- x
    !       |
    !       |
    !       |
    !       z

    ! Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
    call test_function(dim3((512-1)/256 + 1, 512, 512), dim3(256), 0, c_null_ptr, c_loc(du), nx, ny, nz, hx, hy, hz)
    ! Compute Laplacian (1/2) (x(x-1) + y(y-1) + z(z-1)) = 3 for all interior points
    call laplacian(dim3((512-1)/8 + 1,(512-1)/8 + 1,(512-1)/8 + 1), dim3(8,8,8),0, c_null_ptr, &
                   c_loc(df), c_loc(du), c_loc(dcoeffs_x), c_loc(dcoeffs_y), c_loc(dcoeffs_z), &
                   nx, ny, nz)

    call hipCheck(hipDeviceSynchronize())

    ! Transfer data back to host memory
    call hipCheck(hipMemcpy(out, df, nx*ny*nz, hipMemcpyDeviceToHost))

    expected_f =3.
    count = 0
    do k = 5,ny-4
      do j = 5,nx-4
         do i = 5,nz-4
            pos = i + nz * ((j - 1) + nx * (k - 1))
            if (abs(out(pos) - expected_f) / expected_f .gt. tol) count=count+1
         end do
      enddo
    enddo
    if (count .ne. 0) print*, "Correctness test failed. Pointwise error larger than ", tol

    deallocate(hcoeffs_x, hcoeffs_y, hcoeffs_z, out)
end program main
