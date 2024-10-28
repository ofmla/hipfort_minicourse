program fortran_hip
  use iso_c_binding, only: c_int, c_double, c_loc
  implicit none

  interface
     subroutine dxpy_cfcn(N,a,b,out) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr),value :: a, b, out
       integer(c_int), value :: N
     end subroutine
  end interface

  integer(c_int), parameter :: N = 1000000

  real(c_double),pointer,dimension(:) :: a,b,out

  ! Allocate host memory
  allocate(a(N),b(N),out(N))

  ! Initialize host arrays
  a(:) = 1.0
  b(:) = 1.0

  ! launch kernel
  call dxpy_cfcn(N,c_loc(a),c_loc(b),c_loc(out))

  if ( sum(out) .eq. N*2.0 ) then
     print *, "PASSED!"
  else
     print *, "FAILED!"
  endif

  print*, out(1:10)

  ! Deallocate host memory
  deallocate(a,b,out)

end program fortran_hip
