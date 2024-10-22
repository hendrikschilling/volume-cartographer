#ifndef OPSLIST_HPP
#define OPSLIST_HPP

#include <QWidget>

class OpChain;
class Surface;
class QTreeWidget;
class QTreeWidgetItem;
class QComboBox;
class ChunkCache;
namespace z5 {
    class Dataset;
};

namespace Ui
{
class OpsList;
}

class OpsList : public QWidget
{
    Q_OBJECT

public:
    explicit OpsList(QWidget* parent = nullptr);
    ~OpsList();

    void setDataset(z5::Dataset *ds, ChunkCache *cache, float scale);


private slots:
    void onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);

public slots:
    void onOpChainSelected(OpChain *ops);
    void onAppendOpClicked();

signals:
    void sendOpSelected(Surface *surf, OpChain *chain);
    void sendOpChainChanged(OpChain *chain);

private:
    Ui::OpsList* ui;
    QTreeWidget *_tree;
    QComboBox *_add_sel;
    OpChain *_op_chain = nullptr;

    //FIXME currently stored for refinement layer - make this somehow generic ...
    z5::Dataset *_ds = nullptr;
    ChunkCache *_cache = nullptr;
    float _scale = 0.0;
};

#endif  // OPSLIST_HPP
