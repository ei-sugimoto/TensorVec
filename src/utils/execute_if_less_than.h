const int threshold_matrix_size = 16;

template <typename Func>
void execute_if_less_than(int matirxSize, Func func)
{
    if (matirxSize < threshold_matrix_size)
    {
        func();
    }
}