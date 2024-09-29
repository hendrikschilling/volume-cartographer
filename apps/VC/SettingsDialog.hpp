#pragma once

#include "ui_VCSettings.h"

namespace ChaoVis
{

class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        SettingsDialog(QWidget* parent = nullptr);

        static std::vector<int> expandSettingToIntRange(const QString& setting);

    protected slots:
        void accept() override;
};

}