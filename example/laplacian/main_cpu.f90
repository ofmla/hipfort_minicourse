program main
    implicit none
    integer, parameter :: nx=512, ny=512, nz=512
    real, parameter :: tol=3.e-1
    ! https://web.media.mit.edu/~crtaylor/calculator.html
    ! locations of sampled points: -4,-3,-2,-1,0,1,2,3,4
    ! derivative order: 2
    ! fractions are break down into its simplest form
    real, parameter :: c0 = -205./72., c1 =  8./5.,  c2 = -1./5.
    real, parameter :: c3 =  8./315.,  c4 = -1./560.
    real, allocatable :: u(:,:,:), f(:,:,:) 
    real :: hx, hy, hz, expected_f
    integer :: i, j, k, count, pos

    hx = 1.0 / (nx - 1);
    hy = 1.0 / (ny - 1);
    hz = 1.0 / (nz - 1);

    allocate(u(nz,nx,ny), f(nz,nx,ny))

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
    call test_function_kernel(u, nx, ny, nz, hx, hy, hz)

    ! Compute Laplacian (1/2) (x(x-1) + y(y-1) + z(z-1)) = 3 for all interior points
    call laplacian(f, u, nx, ny, nz, hx, hy, hz)
    
    expected_f =3.
    count = 0; 
    do k = 5,ny-4
      do j = 5,nx-4
         do i = 5,nz-4
            if (abs(f(i,j,k) - expected_f) / expected_f .gt. tol) count=count+1
         end do
      enddo
    enddo
    if (count .ne. 0) print*, "Correctness test failed. Pointwise error larger than ", tol

    deallocate (u,f)

    contains

subroutine test_function_kernel(u, nx, ny, nz, hx, hy, hz)
  implicit none
  integer, intent(in) :: nx, ny, nz
  real, intent(in) :: hx, hy, hz
  real, dimension(nz,nx,ny), intent (out) :: u
  integer :: i, j, k
  real:: c, x, y, z, lx, ly, lz
  integer :: pos
  
  c = 0.5
  lx = nx * hx
  ly = ny * hy
  lz = nz * hz
  
  ! Loop over each (i, j, k) point
  do k = 1,ny
    do j = 1,nx
       do i = 1,nz
          pos = i + nz * ((j - 1) + nx * (k - 1))
          ! Compute x, y, z based on grid indices
          y = (k - 1) * hy
          x = (j - 1) * hx
          z = (i - 1) * hz
          ! Compute the test function and store it in u array
          u(i,j,k) = c * (x * (x - lx) + y * (y - ly) + z * (z - lz))
       end do
    enddo
  enddo
end subroutine test_function_kernel
  
subroutine laplacian(f, u, nx, ny, nz, hx, hy, hz)
  implicit none
  integer, intent(in) :: nx, ny, nz
  real, intent(in) :: hx, hy, hz
  real, dimension(nz,nx,ny), intent (in) :: u
  real, dimension(nz,nx,ny), intent (out) :: f
  integer :: i, j, k
  real :: c0z, c1z, c2z, c3z, c4z
  real :: c0y, c1y, c2y, c3y, c4y
  real :: c0x, c1x, c2x, c3x, c4x

  c0z = c0/hz/hz; c0x = c0/hx/hx; c0y = c0/hy/hy
  c1z = c1/hz/hz; c1x = c1/hx/hx; c1y = c1/hy/hy
  c2z = c2/hz/hz; c2x = c2/hx/hx; c2y = c2/hy/hy
  c3z = c3/hz/hz; c3x = c3/hx/hx; c3y = c3/hy/hy
  c4z = c4/hz/hz; c4x = c4/hx/hx; c4y = c4/hy/hy
  
  do k=5,ny-4
    do j=5,nx-4
      do i=5,nz-4
        f(i,j,k) = (c0z + c0x +c0y) * u(i,j,k) + &
                   c1z * (u(i+1,j,k) + u(i-1,j,k)) + &
                   c2z * (u(i+2,j,k) + u(i-2,j,k)) + &
                   c3z * (u(i+3,j,k) + u(i-3,j,k)) + &
                   c4z * (u(i+4,j,k) + u(i-4,j,k)) + &
                   c1x * (u(i,j+1,k) + u(i,j-1,k)) + &
                   c2x * (u(i,j+2,k) + u(i,j-2,k)) + &
                   c3x * (u(i,j+3,k) + u(i,j-3,k)) + &
                   c4x * (u(i,j+4,k) + u(i,j-4,k)) + &
                   c1y * (u(i,j,k+1) + u(i,j,k-1)) + &
                   c2y * (u(i,j,k+2) + u(i,j,k-2)) + &
                   c3y * (u(i,j,k+3) + u(i,j,k-2)) + &
                   c4y * (u(i,j,k+4) + u(i,j,k-4))
      end do
    end do
  end do
end subroutine laplacian
end program main
