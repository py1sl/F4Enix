"""This module is related to the parsing of D1S-UNED meshinfo files."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BYTE = np.byte
CHAR = np.char
INT = np.int32
FLOAT = np.float64
LONG = np.int64
NUMBER_OF_EXPECTED_VALUES = 11  # Number of values recorded for each particle
NEUTRON_INDICATOR = 8  # Neutron indicator in the packed variable


@dataclass
class _SurfaceParameters:
    id: int
    info: int
    type: int
    num_parameters: int
    parameters: list[int]


@dataclass
class _FileParameters:
    np1: int  # Number of histories of the simulation, given as a negative number
    nrss: int  # Number of tracks recorded
    nrcd: int  # Number of values recorded for each particle, it should be 11
    njsw: int  # Number of surfaces in JASW
    niss: int  # Number of different histories that reached the SSW surfaces
    niwr: int  # Number of cells in RSSA file
    mipts: int  # Source particle type
    kjaq: int  # Flag for macrobodies surfaces
    surfaces: list[_SurfaceParameters]  # List with surface ids that appear in the file


class RSSA:
    def __init__(self, path: Path):
        """Representation of a RSSA file.

        Parameters
        ----------
        path: Path
            Path to the RSSA file.

        Attributes
        ----------
        path: Path
            Path to the RSSA file.
        parameters: _FileParameters
            Parameters extracted from the RSSA file header.

            np1   # Number of histories of the simulation, given as a negative number
            nrss  # Number of tracks recorded
            nrcd  # Number of values recorded for each particle, it should be 11
            njsw  # Number of surfaces in JASW
            niss  # Number of different histories that reached the SSW surfaces
            niwr  # Number of cells in RSSA file
            mipts  # Source particle type
            kjaq  # Flag for macrobodies surfaces
            surfaces
        tracks: pl.DataFrame
            DataFrame containing the tracks recorded in the RSSA file.

            Each row of the table has 11 values
            0 a,  # History number of the particle, negative if uncollided
            1 b,  # Packed variable, the sign is the sign of the third direction cosine
                  # starts with 8 = neutron, 16 = photon
            2 wgt,
            3 erg,
            4 tme,
            5 x,
            6 y,
            7 z,
            8 u,  # Particle direction cosine with X-axis
            9 v,  # Particle direction cosine with Y-axis, to calculate w (Z-axis) use
                  # the sign from b
            10 c  # Surface id

        Examples
        --------
        >>> from f4enix.output.rssa import RSSA
        ... my_rssa = RSSA('small_cyl.w')
        ... print(my_rssa)
        RSSA file small_cyl.w was recorded using the following surfaces:
          Surface ID: 1, type: 1
        The total number of tracks recorded is 72083.
        Neutrons: 72083 photons: 0.
        The simulation that produced this RSSA run 100000 histories.
        The amount of independent histories that reached the RSSA surfaces was 70797.
        """
        self.path = path
        with open(path, "rb") as infile:
            self.parameters = _parse_header(infile)
            self.tracks = _parse_tracks(infile)

        # Modify the value of "b" for fast filtering of neutrons and photons
        self.tracks = self.tracks.with_columns(
            (pl.col("b").abs() / (10 ** pl.col("b").abs().log10().floor()))
            .cast(int)
            .alias("b")
        )

    def __repr__(self) -> str:
        return self.get_summary()

    def __str__(self) -> str:
        return self.get_summary()

    def get_summary(self) -> str:
        """Returns a summary of the RSSA file."""
        summary = f"RSSA file {self.path.name} was recorded using the following"
        summary += " surfaces:\n"
        for surface in self.parameters.surfaces:
            summary += f"  Surface ID: {surface.id}, type: {surface.type}\n"

        summary += f"The total number of tracks recorded is {self.parameters.nrss}.\n"
        summary += f"Neutrons: {self.neutron_tracks.shape[0]}"
        summary += f" photons: {self.photon_tracks.shape[0]}, "

        summary += "The simulation that produced this RSSA run "
        summary += f"{abs(self.parameters.np1)} histories.\n"
        summary += "The amount of independent histories that reached the RSSA surfaces "
        summary += f"was {self.parameters.niss}.\n"
        return summary

    @property
    def neutron_tracks(self) -> pl.DataFrame:
        """Returns the neutron tracks from the RSSA file."""
        return self.tracks.filter(pl.col("b") == NEUTRON_INDICATOR)

    @property
    def photon_tracks(self) -> pl.DataFrame:
        """Returns the photon tracks from the RSSA file."""
        return self.tracks.filter(pl.col("b") != NEUTRON_INDICATOR)

    def plot_tracks_on_cyl(self, bin_width: int | float = 10) -> None:
        raise NotImplementedError


def get_2d_grid_of_weights(
    df: pl.DataFrame,
    x_bins: Sequence[float] | pl.Series,
    y_bins: Sequence[float] | pl.Series,
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    # Ensure bins are sorted for search_sorted function
    x_bins = pl.Series("x_bins", x_bins).sort()
    y_bins = pl.Series("y_bins", y_bins).sort()

    # Remove points outside of the bins
    filtered_df = (
        df.lazy()
        .select([x_col, y_col, "wgt"])
        .filter(
            pl.col(x_col).is_between(x_bins[0], x_bins[-1], closed="left"),
            pl.col(y_col).is_between(y_bins[0], y_bins[-1], closed="left"),
        )
        .collect()
    )
    grid = (
        filtered_df.lazy()
        # Find the bin indices for each point
        .with_columns(
            (x_bins.search_sorted(filtered_df[x_col], side="right") - 1).alias("bin_x"),
            (y_bins.search_sorted(filtered_df[y_col], side="right") - 1).alias("bin_y"),
        )
        # Group by the bin indices and sum the weights
        .group_by(["bin_x", "bin_y"])
        .agg(pl.col("wgt").sum().alias("wgt"))
        .collect()
    )
    raster = _get_raster(grid, x_bins, y_bins)

    return raster


def _get_raster(
    grid: pl.DataFrame,
    x_bins: pl.Series,
    y_bins: pl.Series,
) -> np.ndarray:
    # The dimensions of our grid are determined by the number of bins
    num_y_bins = len(y_bins) - 1
    num_x_bins = len(x_bins) - 1
    raster = np.zeros((num_y_bins, num_x_bins), dtype=np.float64)

    # Extract columns to NumPy and use advanced indexing to fill the raster
    bin_x = grid.get_column("bin_x").to_numpy()
    bin_y = grid.get_column("bin_y").to_numpy()
    wgt = grid.get_column("wgt").to_numpy()

    # The bin indices from Polars directly correspond to the raster indices
    raster[bin_y, bin_x] = wgt

    return raster


def calculate_areas(
    x_bins: Sequence[float] | pl.Series, y_bins: Sequence[float] | pl.Series
) -> np.ndarray:
    """Calculate the areas of the bins in a 2D histogram.

    Parameters
    ----------
    x_bins: Sequence[float] | pl.Series
        The x-axis bin edges.
    y_bins: Sequence[float] | pl.Series
        The y-axis bin edges.

    Returns
    -------
    np.ndarray
        A 2D array with the areas of each bin.
    """
    x_edges = np.array(x_bins)
    y_edges = np.array(y_bins)
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    return np.outer(dy, dx)


def _parse_header(infile: BinaryIO) -> _FileParameters:
    first_record = _read_fortran_record(infile)
    # The first line of the file with information like the code version, date and title
    formatted_record_id = first_record.tobytes().decode("UTF-8")
    if "d1suned" in formatted_record_id:
        _last_dump = np.frombuffer(first_record[-4:], INT)
    elif "SF_00001" in formatted_record_id:
        _header = _read_fortran_record(infile)  # code version and other info
    else:
        raise NotImplementedError(
            f"The code that generated this RSSA file has not been implemented"
            f" in this parser, see the code here: {formatted_record_id}..."
        )

    second_record = _read_fortran_record(infile)
    np1 = np.frombuffer(second_record, LONG, 1, 0)[0]
    nrss = np.frombuffer(second_record, LONG, 1, 8)[0]
    nrcd = np.frombuffer(second_record, INT, 1, 16)[0]
    njsw = np.frombuffer(second_record, INT, 1, 20)[0]
    niss = np.frombuffer(second_record, LONG, 1, 24)[0]
    if abs(nrcd) != NUMBER_OF_EXPECTED_VALUES:
        raise NotImplementedError(
            "The amount of values recorded for each particle should be 11 instead of"
            f" {nrcd}..."
        )

    if np1 < 0:
        third_record = _read_fortran_record(infile)
        niwr, mipts, kjaq = np.frombuffer(third_record, INT, 3)
    else:
        raise NotImplementedError("The np1 value is not negative...")

    surfaces = []
    for _ in range(njsw):
        data = _read_fortran_record(infile)
        surf_id = np.frombuffer(data, INT, 1, 0)[0]
        surf_info = np.frombuffer(data, INT, 1, 4)[0] if kjaq == 1 else -1
        surf_type = np.frombuffer(data, INT, 1, 8)[0]
        num_parameters = np.frombuffer(data, INT, 1, 12)[0]
        parameters = np.frombuffer(data, INT, offset=16).tolist()
        surfaces.append(
            _SurfaceParameters(
                id=surf_id,
                info=surf_info,
                type=surf_type,
                num_parameters=num_parameters,
                parameters=parameters,
            )
        )

    # we read any extra records as determined by njsw+niwr...
    # no known case of their actual utility
    for _j in range(njsw, njsw + niwr):
        _read_fortran_record(infile)
        raise NotImplementedError(
            "njsw + niwr values are bigger than njsw, behavior not explained"
        )

    # Summary record
    _data = _read_fortran_record(infile)
    # Summary record not processed, its information does not interest us for now

    return _FileParameters(
        np1=np1,  # Number of histories of the simulation, given as a negative number
        nrss=nrss,  # Number of tracks recorded
        nrcd=nrcd,  # Number of values recorded for each particle, it should be 11
        njsw=njsw,  # Number of surfaces in JASW
        niss=niss,  # Number of different histories that reached the SSW surfaces
        niwr=niwr,  # Number of cells in RSSA file
        mipts=mipts,  # Source particle type
        kjaq=kjaq,  # Flag for macrobodies surfaces
        surfaces=surfaces,
    )


def _parse_tracks(file: BinaryIO) -> pl.DataFrame:
    # Read the whole remaining of the file at once, store all the bytes as a 1D np array
    data = np.fromfile(file, BYTE)

    # Reshape the array so each index holds the information of a single particle
    # we can do this because we know that the particle records have always the same
    # length, 96 bytes
    data = data.reshape(-1, 96)

    # Remove the first and last 4 bytes, these are two integers that tell the record is
    # 88 bytes long
    data = data[:, 4:-4]

    # Convert the array into a 1D array of float numbers instead of simply bytes
    data = np.frombuffer(data.flatten(), FLOAT)

    # Reshape the array so each index holds the information of a single particle
    # all the data is already converted from bytes to floats
    data = data.reshape(-1, 11)

    return pl.DataFrame(
        data,
        schema={
            "a": int,
            "b": int,
            "wgt": float,
            "erg": float,
            "tme": float,
            "x": float,
            "y": float,
            "z": float,
            "u": float,
            "v": float,
            "c": int,
        },
    )


def _read_fortran_record(infile: BinaryIO):
    count_1 = np.fromfile(infile, INT, 1)[0]
    data = np.fromfile(infile, np.byte, count_1)
    count_2 = np.fromfile(infile, INT, 1)[0]
    if count_1 != count_2:
        raise ValueError(
            "The integers that go before and after the Fortran record are not equal..."
        )
    return data
