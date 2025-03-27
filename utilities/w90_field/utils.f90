module utils
    implicit none
    public :: LoadSystem, applyField, writeOutput

    private
    character(len=:), allocatable  :: FileName
    integer                        :: nFock, mSize
    integer, allocatable           :: Degen(:)
    real(8)                        :: Rn(3, 3)
    integer, allocatable           :: iRn(:,:)
    complex(8), allocatable        :: H(:,:,:)
    complex(8), allocatable        :: Rhop(:,:,:,:) ! position amtrix elements
    real(8)                        :: Eamp  ! Electric field amplitute
    integer                        :: center

contains

    subroutine LoadArguments()
        implicit none
        integer N, argCount
        character(len=32) :: Eamp_str

        ! Get number of command-line arguments
        argCount = command_argument_count()
        if (argCount < 2) then
            print*, "Usage: ./program <filename> <Eamp>"
            stop
        end if

        ! Read filename
        call get_command_argument(1, length=N)
        allocate(character(N) :: FileName)
        call get_command_argument(1, FileName)

        ! Read electric field argument
        call get_command_argument(2, Eamp_str)
        read(Eamp_str, *) Eamp

        print*, "Filename: ", FileName
        print*, "Electric Field: ", Eamp
    end subroutine LoadArguments

    subroutine LoadSystem()
        integer :: fp, ii, jj, i, j
        real(8) :: R, Im
        real(8) :: a1, a1j, a2, a2j, a3, a3j

        ! Read filename and electric field from command line
        call LoadArguments()

        ! Open file
        open(action='read', file=FileName, newunit=fp)
            read(fp, *)
            read(fp, *) Rn(1, :)
            read(fp, *) Rn(2, :)
            read(fp, *) Rn(3, :)
            read(fp, *) mSize
            read(fp, *) nFock 
       
            ! Allocate arrays
            allocate(H(nFock, mSize, mSize))
            allocate(Rhop(3, nFock, mSize, mSize))
            allocate(Degen(nFock))
            allocate(iRn(nFock, 3))

            ! Read degeneracies
            if ((nFock / 15) > 1) then
                do i = 1, (nFock / 15)
                    read(fp, *) Degen((i - 1) * 15 + 1:(i - 1) * 15 + 15)
                end do
            end if
            read(fp, *) Degen((i - 1) * 15 + 1:(i - 1) * 15 + MOD(nFock, 15))
            read(fp, *)

            ! Read Hamiltonian
            do i = 1, nFock
                read(fp, *) iRn(i, :)
                if ( iRn(i,1) == 0 .and. iRn(i,2) == 0 .and. iRn(i,3) == 0 ) center = i
                do j = 1, mSize * mSize
                    read(fp, *) ii, jj, R, Im
                    H(i, ii, jj) = cmplx(R, Im, 8)
                end do
                if (i < nFock) read(fp, *)
            end do
            read(fp, *)

            ! Read motif
            do i = 1, nFock
                read(fp, *)
                do j = 1, mSize * mSize
                    read(fp, *) ii, jj, a1, a1j, a2, a2j, a3, a3j
                    Rhop(1, i, ii, jj) = cmplx(a1, a1j, 8)
                    Rhop(2, i, ii, jj) = cmplx(a2, a2j, 8)
                    Rhop(3, i, ii, jj) = cmplx(a3, a3j, 8)
                end do
                if (i < nFock) read(fp, *)
            end do
        close(fp)
    end subroutine LoadSystem

subroutine applyField(Eamp)
    implicit none
    real(8), intent(in) :: Eamp
    real(8)             :: E_field(1,3)
    integer             :: i, ii, jj, k, kk
    complex(8)          :: deltaH, braket
    complex(8)          :: R(3), R0(3)
    real(8), parameter  :: r_z(3) = (/ 0.0d0, 0.0d0, 1.0d0 /)  ! (0,0,1) operator

    E_field = Eamp   ! dump, but useful for later generalizations 
    do i = 1, nFock
        do ii = 1, mSize
            do jj = 1, mSize
                ! Compute the perturbation: 
                deltaH = Eamp * Rhop(3, i, ii, jj) ! r_x,y,z enters here on future generalization

                ! Update the Hamiltonian
                H(i, ii, jj) = H(i, ii, jj) - deltaH
            end do
        end do
    end do

    print*, "Electric field applied."
end subroutine applyField

    subroutine writeOutput()
        implicit none
        integer  :: i, ii, jj, j, fp

        call applyField(Eamp)

        ! Open output file
        open(action = 'write', file="field_tb.dat", newunit=fp)

        ! Write initial information
        write(fp, *) 
        write(fp, *) Rn(1, :)
        write(fp, *) Rn(2, :)
        write(fp, *) Rn(3, :)
        write(fp, *) mSize
        write(fp, *) nFock

        ! Write degeneracies
        do i = 1, (nFock / 15)
            write(fp, *) Degen((i - 1)*15 + 1:(i - 1)*15 + 15)
        enddo
        write(fp, *) Degen((i - 1)*15 + 1:(i - 1)*15 + MOD(nFock, 15))
        write(fp, *)

        ! Begin writing Hamiltonian
        do i = 1, nFock
            write(fp, *) iRn(i, :)
            do ii = 1, mSize
                do jj = 1, mSize
                    write(fp, *) ii, jj, real(H(i, ii, jj)), aimag(H(i, ii, jj))
                enddo
            enddo
            if (i < nFock) write(fp, *)
        enddo
        write(fp,*) 
        
        ! Compute and write new Rhop values with <0|z|R>
        do i = 1, nFock
            write(fp, *) iRn(i, :)

            do ii = 1, mSize
                do jj = 1, mSize
                    write(fp, *) ii, jj, real(Rhop(1, i, ii, jj)), aimag(Rhop(1, i, ii, jj)), &
                                       real(Rhop(2, i, ii, jj)), aimag(Rhop(2, i, ii, jj)), &
                                       real(Rhop(3, i, ii, jj)), aimag(Rhop(3, i, ii, jj))
                enddo
            enddo

            if (i < nFock) write(fp, *)
        enddo
        close(fp)

        print*, "Outfile field_tb.dat is ready."
    end subroutine writeOutput
end module