import adios2
import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = "ICs.bp"
    variable_name = filename.split(".bp")[0]   # <-- CHANGE to your actual variable name

    # Initialize ADIOS
    adios = adios2.Adios()
    io = adios.declare_io("reader")

    # Open file
    engine = io.open(filename, adios2.bindings.Mode.Read)

    status = engine.begin_step()
    if status != adios2.bindings.StepStatus.OK:
        raise RuntimeError("No step available in BP file")

    # Inquire variable
    var = io.inquire_variable(variable_name)
    if var is None:
        print("Available variables:")
        print(io.available_variables())
        raise RuntimeError(f"Variable '{variable_name}' not found")

    shape = var.shape()
    print("Variable shape:", shape)

        # Unpack dimensions
    n_loc3, n_loc2, n_loc1, n_fields, nz_block, ny_block, nx_block = shape
    print(f"Meshblocks: {n_loc3} x {n_loc2} x {n_loc1}, fields={n_fields}, block size={nz_block}x{ny_block}x{nx_block}")

    # Total domain size
    Nz = n_loc3 * nz_block
    Ny = n_loc2 * ny_block
    Nx = n_loc1 * nx_block

    # Midplane in global z
    k_global = Nz // 2

    density_slice = np.zeros((Ny, Nx), dtype=np.float64)

    # Field that is being read
    field_idx = 0  # Assuming density is the first field

    for loc3 in range(n_loc3):
        for loc2 in range(n_loc2):
            for loc1 in range(n_loc1):
                local_k = k_global - loc3 * nz_block
                if not (0 <= local_k < nz_block):
                    continue

                # Shape: [n_loc3, n_loc2, n_loc1, n_fields, nz, ny, nx]
                # field_idx = 0 assuming density is the first field
                start = [loc3, loc2, loc1, field_idx, local_k, 0,        0       ]
                count = [1,    1,    1,    1,          1,       ny_block, nx_block]

                var.set_selection([start, count])
                data_block = np.zeros(count, dtype=np.float64)
                engine.get(var, data_block)
                engine.perform_gets()

                j_start = loc2 * ny_block
                j_end   = j_start + ny_block
                i_start = loc1 * nx_block
                i_end   = i_start + nx_block

                density_slice[j_start:j_end, i_start:i_end] = data_block[0, 0, 0, 0, 0, :, :]

    engine.end_step()
    engine.close()

    print(f"Global slice shape: {density_slice.shape}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.imshow(density_slice.T, origin="lower", aspect="auto")
    plt.colorbar(label="Density")
    plt.title(f"Midplane z-slice of full simulation domain (k={k_global})")
    plt.xlabel("i (global)")
    plt.ylabel("j (global)")
    plt.tight_layout()
    plt.savefig("density_slice.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
