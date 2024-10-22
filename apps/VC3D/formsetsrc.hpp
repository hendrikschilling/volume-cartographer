#ifndef FORMSETSRC_HPP
#define FORMSETSRC_HPP

#include <QWidget>
#include "OpChain.hpp"

namespace Ui
{
class FormSetSrc;
}

class QComboBox;

class FormSetSrc : public QWidget
{
    Q_OBJECT

public:
    explicit FormSetSrc(Surface *op, QWidget* parent = nullptr);
    ~FormSetSrc();
    
private slots:
    void onAlgoIdxChanged(int index);
    
signals:
    void sendOpChainChanged(OpChain *chain);

private:
    Ui::FormSetSrc* ui;
    
    OpChain *_chain;
    QComboBox *_combo;
};

#endif  // FORMSETSRC_HPP
