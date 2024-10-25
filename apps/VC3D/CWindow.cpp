// CWindow.cpp
// Chao Du 2014 Dec
#include "CWindow.hpp"

#include <QKeySequence>
#include <QProgressBar>
#include <QSettings>
#include <QMdiArea>
#include <opencv2/imgproc.hpp>

#include "CVolumeViewer.hpp"
#include "UDataManipulateUtils.hpp"
#include "SettingsDialog.hpp"
#include "CSurfaceCollection.hpp"
#include "OpChain.hpp"
#include "opslist.hpp"
#include "opssettings.hpp"


#include "vc/core/types/Color.hpp"
#include "vc/core/types/Exceptions.hpp"
#include "vc/core/util/Iteration.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/meshing/OrderedPointSetMesher.hpp"
#include "vc/segmentation/LocalResliceParticleSim.hpp"
#include "vc/segmentation/OpticalFlowSegmentation.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

namespace vc = volcart;
namespace vcs = volcart::segmentation;
using namespace ChaoVis;
using qga = QGuiApplication;

// Constructor
CWindow::CWindow() :
    fVpkg(nullptr)
{
    const QSettings settings("VC.ini", QSettings::IniFormat);
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    // setAttribute(Qt::WA_DeleteOnClose);

    //TODO make configurable
    chunk_cache = new ChunkCache(10e9);
    
    _surf_col = new CSurfaceCollection();
    
    _surf_col->setSurface("manual plane", new PlaneSurface({2000,2000,2000},{1,1,1}));
    _surf_col->setSurface("xy plane", new PlaneSurface({2000,2000,2000},{0,0,1}));
    _surf_col->setSurface("xz plane", new PlaneSurface({2000,2000,2000},{0,1,0}));
    _surf_col->setSurface("yz plane", new PlaneSurface({2000,2000,2000},{1,0,0}));
    
    // create UI widgets
    CreateWidgets();

    // create menu
    CreateActions();
    CreateMenus();
    UpdateRecentVolpkgActions();

    if (QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark) {
        // stylesheet
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(55, 80, 170), stop:0.8 rgb(225, 90, 80), stop:1 rgb(225, 150, 0)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(235, 180, 30); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(55, 55, 55); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(60, 60, 75); }"
            "QTabBar::tab { background: rgb(60, 60, 75); }"
            "QWidget#tabSegment { background: rgb(55, 55, 55); }";
        setStyleSheet(style);
    } else {
        // stylesheet
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(85, 110, 200), stop:0.8 rgb(255, 120, 110), stop:1 rgb(255, 180, 30)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(255, 200, 50); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(245, 245, 255); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(205, 210, 240); }"
            "QTabBar::tab { background: rgb(205, 210, 240); }"
            "QWidget#tabSegment { background: rgb(245, 245, 255); }"
            "QRadioButton:disabled { color: gray; }";
        setStyleSheet(style);
    }

    // Restore geometry / sizes
    const QSettings geometry;
    if (geometry.contains("mainWin/geometry")) {
        restoreGeometry(geometry.value("mainWin/geometry").toByteArray());
    }
    if (geometry.contains("mainWin/state")) {
        restoreState(geometry.value("mainWin/state").toByteArray());
    }

    // If enabled, auto open the last used volpkg
    if (settings.value("volpkg/auto_open", false).toInt() != 0) {

        QStringList files = settings.value("volpkg/recent").toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            Open(files[0]);
        }
    }
}

// Destructor
CWindow::~CWindow(void)
{
}

CVolumeViewer *CWindow::newConnectedCVolumeViewer(std::string show_surf, QMdiArea *mdiArea)
{
    auto volView = new CVolumeViewer(_surf_col, mdiArea);
    QMdiSubWindow *win = mdiArea->addSubWindow(volView);
    win->setWindowTitle(show_surf.c_str());
    volView->setCache(chunk_cache);
    connect(this, &CWindow::sendVolumeChanged, volView, &CVolumeViewer::OnVolumeChanged);
    connect(_surf_col, &CSurfaceCollection::sendSurfaceChanged, volView, &CVolumeViewer::onSurfaceChanged);
    connect(_surf_col, &CSurfaceCollection::sendPOIChanged, volView, &CVolumeViewer::onPOIChanged);
    connect(_surf_col, &CSurfaceCollection::sendIntersectionChanged, volView, &CVolumeViewer::onIntersectionChanged);
    connect(volView, &CVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked);
    
    volView->setSurface(show_surf);
    
    _viewers.push_back(volView);
    
    return volView;
}

void CWindow::setVolume(std::shared_ptr<volcart::Volume> newvol)
{
    currentVolume = newvol;

    //FIXME currently hardcoded to 0.5
    wOpsList->setDataset(currentVolume->zarrDataset(1), chunk_cache, 0.5);
    
    int w = currentVolume->sliceWidth();
    int h = currentVolume->sliceHeight();
    int d = currentVolume->numSlices();
    
    // onVolumeClicked({0,0},{w/2,h/2,d/2});
    
    onManualPlaneChanged();
    sendVolumeChanged(currentVolume);
}

// Create widgets
void CWindow::CreateWidgets(void)
{
    QSettings settings("VC.ini", QSettings::IniFormat);

    // add volume viewer
    auto aWidgetLayout = new QVBoxLayout;
    ui.tabSegment->setLayout(aWidgetLayout);
    
    QMdiArea *mdiArea = new QMdiArea(ui.tabSegment);
    aWidgetLayout->addWidget(mdiArea);
    
    newConnectedCVolumeViewer("manual plane", mdiArea);
    newConnectedCVolumeViewer("seg xz", mdiArea);
    newConnectedCVolumeViewer("seg yz", mdiArea);
    newConnectedCVolumeViewer("xy plane", mdiArea);
    newConnectedCVolumeViewer("xz plane", mdiArea);
    newConnectedCVolumeViewer("yz plane", mdiArea);
    newConnectedCVolumeViewer("segmentation", mdiArea)->setIntersects({"seg xz","seg yz"});
    mdiArea->tileSubWindows();

    treeWidgetSurfaces = this->findChild<QTreeWidget*>("treeWidgetSurfaces");
    auto dockWidgetOpList = this->findChild<QDockWidget*>("dockWidgetOpList");
    auto dockWidgetOpSettings = this->findChild<QDockWidget*>("dockWidgetOpSettings");
    wOpsList = new OpsList(dockWidgetOpList);
    dockWidgetOpList->setWidget(wOpsList);
    wOpsSettings = new OpsSettings(dockWidgetOpSettings);
    dockWidgetOpSettings->setWidget(wOpsSettings);

    connect(treeWidgetSurfaces, &QTreeWidget::currentItemChanged, this, &CWindow::onSurfaceSelected);
    connect(this, &CWindow::sendOpChainSelected, wOpsList, &OpsList::onOpChainSelected);
    connect(wOpsList, &OpsList::sendOpSelected, wOpsSettings, &OpsSettings::onOpSelected);

    connect(wOpsList, &OpsList::sendOpChainChanged, this, &CWindow::onOpChainChanged);
    connect(wOpsSettings, &OpsSettings::sendOpChainChanged, this, &CWindow::onOpChainChanged);

    // new and remove path buttons
    // connect(ui.btnNewPath, SIGNAL(clicked()), this, SLOT(OnNewPathClicked()));
    // connect(ui.btnRemovePath, SIGNAL(clicked()), this, SLOT(OnRemovePathClicked()));

    // TODO CHANGE VOLUME LOADING; FIRST CHECK FOR OTHER VOLUMES IN THE STRUCTS
    volSelect = this->findChild<QComboBox*>("volSelect");
    connect(
        volSelect, &QComboBox::currentIndexChanged, [this](const int& index) {
            vc::Volume::Pointer newVolume;
            try {
                newVolume = fVpkg->volume(volSelect->currentData().toString().toStdString());
            } catch (const std::out_of_range& e) {
                QMessageBox::warning(this, "Error", "Could not load volume.");
                return;
            }
            setVolume(newVolume);
        });

    assignVol = this->findChild<QPushButton*>("assignVol");

    // Set up the status bar
    statusBar = this->findChild<QStatusBar*>("statusBar");

    //new location input
    lblLoc[0] = this->findChild<QLabel*>("sliceX");
    lblLoc[1] = this->findChild<QLabel*>("sliceY");
    lblLoc[2] = this->findChild<QLabel*>("sliceZ");
    
    spNorm[0] = this->findChild<QDoubleSpinBox*>("dspNX");
    spNorm[1] = this->findChild<QDoubleSpinBox*>("dspNY");
    spNorm[2] = this->findChild<QDoubleSpinBox*>("dspNZ");
    
    for(int i=0;i<3;i++)
        spNorm[i]->setRange(-10,10);
    
    connect(spNorm[0], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[1], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[2], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
}

// Create menus
void CWindow::CreateMenus(void)
{
    // "Recent Volpkg" menu
    fRecentVolpkgMenu = new QMenu(tr("Open &recent volpkg"), this);
    fRecentVolpkgMenu->setEnabled(false);
    for (auto& action : fOpenRecentVolpkg)
    {
        fRecentVolpkgMenu->addAction(action);
    }

    fFileMenu = new QMenu(tr("&File"), this);
    fFileMenu->addAction(fOpenVolAct);
    fFileMenu->addMenu(fRecentVolpkgMenu);
    fFileMenu->addSeparator();
    fFileMenu->addAction(fSettingsAct);
    fFileMenu->addSeparator();
    fFileMenu->addAction(fExitAct);

    fEditMenu = new QMenu(tr("&Edit"), this);

    fViewMenu = new QMenu(tr("&View"), this);
    fViewMenu->addAction(findChild<QDockWidget*>("dockWidgetVolumes")->toggleViewAction());

    fHelpMenu = new QMenu(tr("&Help"), this);
    fHelpMenu->addAction(fKeybinds);
    fFileMenu->addSeparator();

    QSettings settings("VC.ini", QSettings::IniFormat);
    if (settings.value("internal/debug", 0).toInt() == 1) {
        fHelpMenu->addAction(fPrintDebugInfo);
        fFileMenu->addSeparator();
    }

    fHelpMenu->addAction(fAboutAct);

    menuBar()->addMenu(fFileMenu);
    menuBar()->addMenu(fEditMenu);
    menuBar()->addMenu(fViewMenu);
    menuBar()->addMenu(fHelpMenu);
}

// Create actions
void CWindow::CreateActions(void)
{
    fOpenVolAct = new QAction(style()->standardIcon(QStyle::SP_DialogOpenButton), tr("&Open volpkg..."), this);
    connect(fOpenVolAct, SIGNAL(triggered()), this, SLOT(Open()));
    fOpenVolAct->setShortcut(QKeySequence::Open);

    for (auto& action : fOpenRecentVolpkg)
    {
        action = new QAction(this);
        action->setVisible(false);
        connect(action, &QAction::triggered, this, &CWindow::OpenRecent);
    }

    fSettingsAct = new QAction(tr("Settings"), this);
    connect(fSettingsAct, SIGNAL(triggered()), this, SLOT(ShowSettings()));

    fExitAct = new QAction(style()->standardIcon(QStyle::SP_DialogCloseButton), tr("E&xit..."), this);
    connect(fExitAct, SIGNAL(triggered()), this, SLOT(close()));

    fKeybinds = new QAction(tr("&Keybinds"), this);
    connect(fKeybinds, SIGNAL(triggered()), this, SLOT(Keybindings()));

    fAboutAct = new QAction(tr("&About..."), this);
    connect(fAboutAct, SIGNAL(triggered()), this, SLOT(About()));
}

void CWindow::UpdateRecentVolpkgActions()
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    QStringList files = settings.value("volpkg/recent").toStringList();
    if (files.isEmpty()) {
        return;
    }

    // The automatic conversion to string list from the settings, (always?) adds an
    // empty entry at the end. Remove it if present.
    if (files.last().isEmpty()) {
        files.removeLast();
    }

    const int numRecentFiles = qMin(files.size(), static_cast<int>(MAX_RECENT_VOLPKG));

    for (int i = 0; i < numRecentFiles; ++i) {
        // Replace "&" with "&&" since otherwise they will be hidden and interpreted
        // as mnemonics
        QString fileName = QFileInfo(files[i]).fileName();
        fileName.replace("&", "&&");
        QString path = QFileInfo(files[i]).canonicalPath();

        if (path == "."){
            path = tr("Directory not available!");
        } else {
            path.replace("&", "&&");
        }

        QString text = tr("&%1 | %2 (%3)").arg(i + 1).arg(fileName).arg(path);
        fOpenRecentVolpkg[i]->setText(text);
        fOpenRecentVolpkg[i]->setData(files[i]);
        fOpenRecentVolpkg[i]->setVisible(true);
    }

    for (int j = numRecentFiles; j < MAX_RECENT_VOLPKG; ++j) {
        fOpenRecentVolpkg[j]->setVisible(false);
    }

    fRecentVolpkgMenu->setEnabled(numRecentFiles > 0);
}

void CWindow::UpdateRecentVolpkgList(const QString& path)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    QStringList files = settings.value("volpkg/recent").toStringList();
    const QString pathCanonical = QFileInfo(path).absoluteFilePath();
    files.removeAll(pathCanonical);
    files.prepend(pathCanonical);

    while(files.size() > MAX_RECENT_VOLPKG) {
        files.removeLast();
    }

    settings.setValue("volpkg/recent", files);

    UpdateRecentVolpkgActions();
}

void CWindow::RemoveEntryFromRecentVolpkg(const QString& path)
{
    QSettings settings("VC.ini", QSettings::IniFormat);
    QStringList files = settings.value("volpkg/recent").toStringList();
    files.removeAll(path);
    settings.setValue("volpkg/recent", files);

    UpdateRecentVolpkgActions();
}

// Asks User to Save Data Prior to VC.app Exit
void CWindow::closeEvent(QCloseEvent* event)
{
    QSettings settings;
    settings.setValue("mainWin/geometry", saveGeometry());
    settings.setValue("mainWin/state", saveState());

    QMainWindow::closeEvent(event);
}

void CWindow::setWidgetsEnabled(bool state)
{
    this->findChild<QGroupBox*>("grpVolManager")->setEnabled(state);
    this->findChild<QGroupBox*>("grpSeg")->setEnabled(state);
    this->findChild<QGroupBox*>("grpEditing")->setEnabled(state);
}

auto CWindow::InitializeVolumePkg(const std::string& nVpkgPath) -> bool
{
    fVpkg = nullptr;

    try {
        fVpkg = vc::VolumePkg::New(nVpkgPath);
    } catch (const std::exception& e) {
        vc::Logger()->error("Failed to initialize volpkg: {}", e.what());
    }

    if (fVpkg == nullptr) {
        vc::Logger()->error("Cannot open .volpkg: {}", nVpkgPath);
        QMessageBox::warning(
            this, "Error",
            "Volume package failed to load. Package might be corrupt.");
        return false;
    }
    return true;
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (fVpkg == nullptr) {
        setWidgetsEnabled(false);  // Disable Widgets for User
        this->findChild<QLabel*>("lblVpkgName")
            ->setText("[ No Volume Package Loaded ]");
        return;
    }

    setWidgetsEnabled(true);  // Enable Widgets for User

    // show volume package name
    this->findChild<QLabel*>("lblVpkgName")
        ->setText(QString(fVpkg->name().c_str()));

    volSelect->setEnabled(can_change_volume_());
    assignVol->setEnabled(can_change_volume_());

    update();
}

void CWindow::onShowStatusMessage(QString text, int timeout)
{
    statusBar->showMessage(text, timeout);
}

fs::path seg_path_name(const fs::path &path)
{
    std::string name;
    bool store = false;
    for(auto elm : path) {
        if (store)
            name += "/"+elm.string();
        else if (elm == "paths")
            store = true;
    }
    name.erase(0,1);
    return name;
}

// Open volume package
void CWindow::OpenVolume(const QString& path)
{
    QString aVpkgPath = path;
    QSettings settings("VC.ini", QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        aVpkgPath = QFileDialog::getExistingDirectory(
            this, tr("Open Directory"), settings.value("volpkg/default_path").toString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        // Dialog box cancelled
        if (aVpkgPath.length() == 0) {
            vc::Logger()->info("Open .volpkg canceled");
            return;
        }
    }

    // Checks the folder path for .volpkg extension
    auto const extension = aVpkgPath.toStdString().substr(
        aVpkgPath.toStdString().length() - 7, aVpkgPath.toStdString().length());
    if (extension != ".volpkg") {
        QMessageBox::warning(
            this, tr("ERROR"),
            "The selected file is not of the correct type: \".volpkg\"");
        vc::Logger()->error(
            "Selected file is not .volpkg: {}", aVpkgPath.toStdString());
        fVpkg = nullptr;  // Is needed for User Experience, clears screen.
        return;
    }

    // Open volume package
    if (!InitializeVolumePkg(aVpkgPath.toStdString() + "/")) {
        return;
    }

    // Check version number
    if (fVpkg->version() < VOLPKG_MIN_VERSION) {
        const auto msg = "Volume package is version " +
                         std::to_string(fVpkg->version()) +
                         " but this program requires version " +
                         std::to_string(VOLPKG_MIN_VERSION) + "+.";
        vc::Logger()->error(msg);
        QMessageBox::warning(this, tr("ERROR"), QString(msg.c_str()));
        fVpkg = nullptr;
        return;
    }

    fVpkgPath = aVpkgPath;
    setVolume(fVpkg->volume());
    {
        const QSignalBlocker blocker{volSelect};
        volSelect->clear();
    }
    QStringList volIds;
    for (const auto& id : fVpkg->volumeIDs()) {
        volSelect->addItem(
            QString("%1 (%2)").arg(QString::fromStdString(id)).arg(QString::fromStdString(fVpkg->volume(id)->name())),
            QVariant(QString::fromStdString(id)));
    }

    treeWidgetSurfaces->clear();

    for (auto& s : fVpkg->segmentationFiles()) {
        QTreeWidgetItem *item = new QTreeWidgetItem(treeWidgetSurfaces);
        item->setText(0, QString(seg_path_name(s.string()).c_str()));
        item->setData(0, Qt::UserRole, QVariant(s.string().c_str()));
    }

    UpdateRecentVolpkgList(aVpkgPath);
}

void CWindow::CloseVolume(void)
{
    fVpkg = nullptr;
    currentVolume = nullptr;
    UpdateView();
}

// Handle open request
void CWindow::Open(void)
{
    Open(QString());
}

// Handle open request
void CWindow::Open(const QString& path)
{
    CloseVolume();
    OpenVolume(path);
    UpdateView();  // update the panel when volume package is loaded
}

void CWindow::OpenRecent()
{
    auto action = qobject_cast<QAction*>(sender());
    if (action)
        Open(action->data().toString());
}

// Pop up about dialog
void CWindow::Keybindings(void)
{
    // REVISIT - FILL ME HERE
    QMessageBox::information(
        this, tr("Keybindings for Volume Cartographer"),
        tr("Keyboard: \n"
        "------------------- \n"
        "FIXME FIXME FIXME \n"
        "------------------- \n"
        "Ctrl+O: Open Volume Package \n"
        "Ctrl+S: Save Volume Package \n"
        "A,D: Impact Range down/up \n"
        "[, ]: Alternative Impact Range down/up \n"
        "Q,E: Slice scan range down/up (mouse wheel scanning) \n"
        "Arrow Left/Right: Slice down/up by 1 \n"
        "1,2: Slice down/up by 1 \n"
        "3,4: Slice down/up by 5 \n"
        "5,6: Slice down/up by 10 \n"
        "7,8: Slice down/up by 50 \n"
        "9,0: Slice down/up by 100 \n"
        "Ctrl+G: Go to slice (opens dialog to insert slice index) \n"
        "T: Segmentation Tool \n"
        "P: Pen Tool \n"
        "Space: Toggle Curve Visibility \n"
        "C: Alternate Toggle Curve Visibility \n"
        "J: Highlight Next Curve that is selected for computation \n"
        "K: Highlight Previous Curve that is selected for computation \n"
        "F: Return to slice that the currently active tool was started on \n"
        "L: Mark/unmark current slice as anchor (only in Segmentation Tool) \n"
        "Y/Z/V: Evenly space Points on Curve (only in Segmentation Tool) \n"
        "U: Rotate view counterclockwise \n"
        "O: Rotate view clockwise \n"
        "X/I: Reset view rotation back to zero \n"
        "\n"
        "Mouse: \n"
        "------------------- \n"
        "Mouse Wheel: Scroll up/down \n"
        "Mouse Wheel + Alt: Scroll left/right \n"
        "Mouse Wheel + Ctrl: Zoom in/out \n"
        "Mouse Wheel + Shift: Next/previous slice \n"
        "Mouse Wheel + W Key Hold: Change impact range \n"
        "Mouse Wheel + R Key Hold: Follow Highlighted Curve \n"
        "Mouse Wheel + S Key Hold: Rotate view \n"
        "Mouse Left Click: Add Points to Curve in Pen Tool. Snap Closest Point to Cursor in Segmentation Tool. \n"
        "Mouse Left Drag: Drag Point / Curve after Mouse Left Click \n"
        "Mouse Right Drag: Pan slice image\n"
        "Mouse Back/Forward Button: Follow Highlighted Curve \n"
        "Highlighting Segment ID: Shift/(Alt as well as Ctrl) Modifier to jump to Segment start/end."));
}

// Pop up about dialog
void CWindow::About(void)
{
    // REVISIT - FILL ME HERE
    QMessageBox::information(
        this, tr("About Volume Cartographer"),
        tr("Vis Center, University of Kentucky\n\n"
        "Fork: https://github.com/hendrikschilling/volume-cartographer"));
}

void CWindow::ShowSettings()
{
    auto pDlg = new SettingsDialog(this);
    pDlg->exec();
    delete pDlg;
}

auto CWindow::can_change_volume_() -> bool
{
    bool canChange = fVpkg != nullptr && fVpkg->numberOfVolumes() > 1;
    return canChange;
}

// Handle request to step impact range down
void CWindow::onLocChanged(void)
{
    // std::cout << "loc changed!" << "\n";
    
    // sendLocChanged(spinLoc[0]->value(),spinLoc[1]->value(),spinLoc[2]->value());
}

void CWindow::onVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    //current action: move default POI
    if (modifiers & Qt::ControlModifier) {
        //NOTE this comes before the focus poi, so focus is applied by views using these slices
        //FIXME this assumes a single segmentation ... make configurable and cleaner ...
        QuadSurface *segment = dynamic_cast<QuadSurface*>(surf);
        if (segment) {
            PlaneSurface *segXZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg xz"));
            PlaneSurface *segYZ = dynamic_cast<PlaneSurface*>(_surf_col->surface("seg yz"));
            
            if (!segXZ)
                segXZ = new PlaneSurface();
            if (!segYZ)
                segYZ = new PlaneSurface();

            //FIXME actually properly use ptr .
            // cv::Vec3f p2;..
            SurfacePointer *ptr = segment->pointer();
            float err = segment->pointTo(ptr, vol_loc, 1.0);
            
            cv::Vec3f p2;
            p2 = segment->coord(ptr, {1,0,0});
            
            segXZ->setOrigin(vol_loc);
            segXZ->setNormal(p2-vol_loc);
            
            p2 = segment->coord(ptr, {0,1,0});
            
            segYZ->setOrigin(vol_loc);
            segYZ->setNormal(p2-vol_loc);
            
            _surf_col->setSurface("seg xz", segXZ);
            _surf_col->setSurface("seg yz", segYZ);
        }
        
        POI *poi = _surf_col->poi("focus");
        
        if (!poi)
            poi = new POI;

        poi->src = surf;
        poi->p = vol_loc;
        poi->n = normal;
        
        _surf_col->setPOI("focus", poi);

        lblLoc[0]->setText(QString::number(poi->p[0]));
        lblLoc[1]->setText(QString::number(poi->p[1]));
        lblLoc[2]->setText(QString::number(poi->p[2]));
    }
    else {
        std::cout << "FIXME do something with regular click" << std::endl;
    }
}

void CWindow::onManualPlaneChanged(void)
{    
    cv::Vec3f normal;
    
    for(int i=0;i<3;i++) {
        normal[i] = spNorm[i]->value();
    }
 
    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf_col->surface("manual plane"));
 
    if (!plane)
        return;
 
    plane->setNormal(normal);
    _surf_col->setSurface("manual plane", plane);
}

void CWindow::onOpChainChanged(OpChain *chain)
{
    _surf_col->setSurface("segmentation", chain);
}

void CWindow::onSurfaceSelected(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    std::string surf_path = current->data(0, Qt::UserRole).toString().toStdString();

    if (!_opchains.count(surf_path)) {
        if (fs::path(surf_path).extension() == ".vcps")
            _opchains[surf_path] = new OpChain(load_quad_from_vcps(surf_path));
        else if (fs::path(surf_path).extension() == ".obj") {
            QuadSurface *quads = load_quad_from_obj(surf_path);
            if (quads)
                _opchains[surf_path] = new OpChain(quads);
        }
    }

    if (_opchains[surf_path]) {
        _surf_col->setSurface("segmentation", _opchains[surf_path]);
        sendOpChainSelected(_opchains[surf_path]);
    }
    else
        std::cout << "ERROR loading " << surf_path << std::endl;
}
