import napari

from napari_spam import SpamLoaderWidget


def main() -> None:
    viewer = napari.Viewer()
    widget = SpamLoaderWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="Spam Loader")
    napari.run()


if __name__ == "__main__":
    main()
