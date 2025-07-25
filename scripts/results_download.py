import gdown
import os


if __name__ == "__main__":
    ids = [
        "17L1sGynt_f04WpiK5d9YHui_c99z1PS9",
        "1ubkj7v5uQRKy3OIkpLd6dlMTFzxa0hoG",
        "12vwz_QsJ9SeBKfWVCR97gfX4ZFej5p5p",
        "1MnxlgOgivebAPo1YcVT_RlNZBeQPCgdG",
        "1WJUUUP3T6GhdX_GwTDk-xkoczbeZoA1p",
        "1CHURhIPsrmZJDrKqK9HZNWu_oHQhLD9i",
    ]
    outputs = [
        "exps.zip",
        "exps_domain_shift.zip",
        "exps_real.zip",
        "exps_safe.zip",
        "exps_thrifty.zip",
        "exps_wo_relabel.zip",
    ]
    for id, output in zip(ids, outputs):
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, output, quiet=False)
        os.system(f"unzip -o {output}")
        os.system(f"rm {output}")
        print(f"Downloaded {output}")
