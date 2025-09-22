import typer
from loguru import logger
import cv2
from pathlib import Path
from .enhancer import EnhancerService

app = typer.Typer(help="Fingerprint enhancement CLI")

@app.command()
def methods():
    """List available enhancement methods."""
    service = EnhancerService()
    for i, m in enumerate(service.list_methods(), 1):
        typer.echo(f"{i}. {m}")

@app.command()
def enhance(input: str, method: str, output: str = "out.bmp"):
    """Enhance a single image with a specific method."""
    service = EnhancerService()
    img = cv2.imread(input)
    if img is None:
        raise typer.BadParameter(f"Cannot read image: {input}")
    result = service.enhance(img, method)
    ok = cv2.imwrite(output, result)
    if not ok:
        raise typer.Exit(code=1)
    typer.echo(f"Saved: {output}")

@app.command()
def enhance_all(input: str, out_dir: str = "outputs"):
    """Enhance an image with all methods. Saves BMP files to out_dir."""
    service = EnhancerService()
    img = cv2.imread(input)
    if img is None:
        raise typer.BadParameter(f"Cannot read image: {input}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = service.enhance_all(img)
    for i, (name, arr) in enumerate(results.items(), 1):
        out_path = Path(out_dir) / f"{i:02d}_{name}.bmp"
        cv2.imwrite(str(out_path), arr)
        typer.echo(f"Saved: {out_path}")
    typer.echo("Done.")

if __name__ == "__main__":
    app()
