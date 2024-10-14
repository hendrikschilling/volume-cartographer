#pragma once

#include <cstddef>
#include <cstdint>

#include <unordered_map>

#include <opencv2/core.hpp>

// #include <QCloseEvent>
// #include <QComboBox>
// #include <QMessageBox>
// #include <QObject>
// #include <QRect>
// #include <QShortcut>
// #include <QSpinBox>
// #include <QThread>
// #include <QTimer>
// #include <QtWidgets>

// #include "BlockingDialog.hpp"
// #include "CBSpline.hpp"
// #include "CXCurve.hpp"
// #include "MathUtils.hpp"
#include "ui_VCMain.h"

// #include "vc/core/types/VolumePkg.hpp"
// #include "vc/segmentation/ChainSegmentationAlgorithm.hpp"

// #include <thread>
// #include <condition_variable>
// #include <atomic>
// #include <SDL2/SDL.h>
// #include <cmath>
// #include <queue>

#define MAX_RECENT_VOLPKG 10

// Volpkg version required by this app
static constexpr int VOLPKG_MIN_VERSION = 6;
static constexpr int VOLPKG_SLICE_MIN_INDEX = 0;

//our own fw declarations
class PlaneCoords;
class CoordGenerator;
class ChunkCache;
class ControlPointSegmentator;

namespace volcart {
    class Volume;
    class VolumePkg;
}

//Qt fw declaration
class QMdiArea;

namespace ChaoVis
{

class CVolumeViewer;
class CSurfaceCollection;

class CWindow : public QMainWindow
{

    Q_OBJECT

public:
    enum SaveResponse : bool { Cancelled, Continue };


signals:
    void sendLocChanged(int x, int y, int z);
    void sendVolumeChanged(std::shared_ptr<volcart::Volume> vol);
    void sendSliceChanged(std::string,CoordGenerator*);

public slots:
    void onShowStatusMessage(QString text, int timeout);
    void onLocChanged(void);
    void onPlaneSliceChanged(void);
    void onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, CoordGenerator *slice, cv::Vec3f slice_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onShiftNormal(cv::Vec3f step);

public:
    CWindow();
    ~CWindow(void);

private:
    void CreateWidgets(void);
    void CreateMenus(void);
    void CreateActions(void);
    void CreateBackend(void);

    void UpdateView(void);

    void UpdateRecentVolpkgActions(void);
    void UpdateRecentVolpkgList(const QString& path);
    void RemoveEntryFromRecentVolpkg(const QString& path);

    CVolumeViewer *newConnectedCVolumeViewer(std::string show_slice, QMdiArea *mdiArea);
    void closeEvent(QCloseEvent* event);

    void setWidgetsEnabled(bool state);

    bool InitializeVolumePkg(const std::string& nVpkgPath);
    void setDefaultWindowWidth(std::shared_ptr<volcart::Volume> volume);

    void OpenVolume(const QString& path);
    void CloseVolume(void);

    static void audio_callback(void *user_data, uint8_t *raw_buffer, int bytes);
    void playPing();

    void setVolume(std::shared_ptr<volcart::Volume> newvol);

private slots:
    void Open(void);
    void Open(const QString& path);
    void OpenRecent();
    void Keybindings(void);
    void About(void);
    void ShowSettings();
private:
    std::shared_ptr<volcart::VolumePkg> fVpkg;
    QString fVpkgPath;
    std::string fVpkgName;

    std::shared_ptr<volcart::Volume> currentVolume;
    int loc[3] = {0,0,0};

    static const int AMPLITUDE = 28000;
    static const int FREQUENCY = 44100;

    // window components
    QMenu* fFileMenu;
    QMenu* fEditMenu;
    QMenu* fViewMenu;
    QMenu* fHelpMenu;
    QMenu* fRecentVolpkgMenu{};

    QAction* fOpenVolAct;
    QAction* fOpenRecentVolpkg[MAX_RECENT_VOLPKG]{};
    QAction* fSavePointCloudAct;
    QAction* fSettingsAct;
    QAction* fExitAct;
    QAction* fKeybinds;
    QAction* fAboutAct;
    QAction* fPrintDebugInfo;

    QComboBox* volSelect;
    QComboBox* previewSelect;
    QPushButton* assignVol;
    
    //TODO abstract these into separate QWidget class?
    QLabel* lblLoc[3];
    QDoubleSpinBox* spNorm[3];

    Ui_VCMainWindow ui;

    QStatusBar* statusBar;

    bool can_change_volume_();
    bool can_change_preview_();
    
    ChunkCache *chunk_cache;
    std::vector<CVolumeViewer*> _viewers;
    CSurfaceCollection *_slices;
};  // class CWindow

}  // namespace ChaoVis
