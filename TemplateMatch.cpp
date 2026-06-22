#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <QPainter>
#include <QDebug>

#include "eprocess.hpp"
#include "ebool_property.hpp"
#include "eint_property.hpp"
#include "erect_property.hpp"
#include "epolygon_property.hpp"
#include "epolygonlist_property.hpp"
#include "eimage_property.hpp"
#include "eutils.hpp"

using namespace cv;
using namespace std;

class TemplateMatch : public EProcess {
    Q_OBJECT
public:
    TemplateMatch(const QString& name, EProperty* parent = 0)
        : EProcess(name, EModule::Ok, tr("Template Match Detection Module"), parent)
        , mbTemplateReady_(false)
    {
        roi_ = new ERectProperty(tr("ROI"), QRect(0, 0, 0, 0),
                                 tr("Template ROI Rect"), this);
        roi_->setMask(TEMPL_MASK);

        first_ = new EBoolProperty(tr("First"), false,
                                   tr("Use first image as template"), this);
        first_->setMask(TEMPL_MASK);

        th_ = new EIntProperty(tr("Thresh"), 80,
                               tr("Match threshold (0-99)"), this,
                               nullptr, nullptr, 0, 99);
        th_->setMask(ALL_MASK);
        //二值化
        //margin_ = new EIntProperty(tr("erzhihua"), 0,
        //                       tr("erzhihua cengdu Max"), this,
        //                            nullptr, nullptr, 0, 99);
       // margin_->setMask(ALL_MASK);
        //

        margin_ = new EIntProperty(tr("Margin"), 0,
                                   tr("Margin around detections"), this,
                                   nullptr, nullptr, 0, 99);
        margin_->setMask(ALL_MASK);

        edgeOverlap_ = new EIntProperty(tr("EdgeOverlap"), 30,
                                        tr("Edge overlap threshold % (0-99)"), this,
                                        nullptr, nullptr, 0, 99);
        edgeOverlap_->setMask(ALL_MASK);

        objList_ = new EPolygonListProperty(tr("Matches"), QVector<QPolygon>(),
                                            tr("Matched regions"), result_);
        objList_->setNoCell(true);
        objList_->setColor(Qt::green);

        diffList_ = new EPolygonListProperty(tr("Diffs"), QVector<QPolygon>(),
                                             tr("Unmatched regions"), result_);
        diffList_->setReadOnly(true);
        diffList_->setNoCell(true);
        diffList_->hide();//隐藏颜色组件下的 Diffs 颜色框，这是做占位符，不影响

        region_ = new EPolygonProperty(tr("Region"), QPolygon(),
                                       tr("Search region in source"), result_);
        region_->setNoCell(true);
        region_->setReadOnly(true);

        image_[0] = new EImageProperty(tr("Template"), QImage(),
                                        tr("Template image"), result_);
        image_[0]->setReadOnly(true);
        image_[0]->hide();
    }

    virtual Type moduleType() override { return Process; }
    virtual bool isUseTempl() const override { return true; }

    virtual void start(bool b) override {
        EModule::start(b);
        mbTemplateReady_ = false;
        tplGradX_.release();
        tplGradY_.release();
        tplEdge_.release();
    }

    // ============================================================
    //  计算 Sobel 梯度特征（X/Y 方向，CV_32F）
    // ============================================================
    static void computeGradient(const cv::Mat& gray, cv::Mat& gradX, cv::Mat& gradY) {
        cv::Mat blur;
        cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
        cv::Mat gx, gy;
        cv::Sobel(blur, gx, CV_32F, 1, 0, 3);
        cv::Sobel(blur, gy, CV_32F, 0, 1, 3);
        // 取绝对值：只保留边缘强度，去除方向符号
        // 配合 TM_CCORR_NORMED（非 CCOEFF）使用，避免均值减法导致的
        // 白区（零梯度）数值不稳定问题
        gradX = cv::abs(gx);
        gradY = cv::abs(gy);
    }

    // ============================================================
    //  从灰度图中提取 ROI 区域的梯度模板
    // ============================================================
    void extractTemplate(const cv::Mat& gray) {
        QRect rc = roi_->getRect();
        cv::Rect roi(rc.x(), rc.y(), rc.width(), rc.height());
        cv::Rect rimg(0, 0, gray.cols, gray.rows);
        roi = roi & rimg;
        if (roi.empty()) roi = rimg;

        cv::Mat roiGray = gray(roi);
        computeGradient(roiGray, tplGradX_, tplGradY_);

        // 提取二值边缘图（用于边缘重叠率计算）
        cv::Canny(roiGray, tplEdge_, 50, 150);

        // 调试输出：模板原图（非梯度，便于查看 ROI 内容）
        if (!isRunning()) {
            QImage qimg;
            matToQImage(roiGray, qimg);
            image_[0]->setImage(qimg);
            image_[0]->show();
        }
    }

    // ============================================================
    //  processTemplate  —  从模板图中提取 ROI 作为匹配模板
    // ============================================================
    virtual void processTemplate(cv::Mat& templ) override {
        if (templ.empty()) return;

        // "First" 模式：跳过模板设置，由 process() 首帧动态提取
        if (first_->getBool() && isRunning()) return;

        cv::Mat gray;
        if (templ.channels() == 3)
            cv::cvtColor(templ, gray, cv::COLOR_BGR2GRAY);
        else
            gray = templ;

        extractTemplate(gray);
        mbTemplateReady_ = true;
    }

    // ============================================================
    //  process  —  Sobel 梯度特征模板匹配
    // ============================================================
    virtual void process(cv::Mat& sImg) override {
        if (sImg.empty()) return;

        cv::Mat gray;
        if (sImg.channels() == 3)
            cv::cvtColor(sImg, gray, cv::COLOR_BGR2GRAY);
        else
            gray = sImg;

        // "First" 模式：首帧动态提取模板
        if (first_->getBool() && !mbTemplateReady_) {
            extractTemplate(gray);
            mbTemplateReady_ = true;
        }

        QVector<QPolygon> matches;

        if (!tplGradX_.empty() && tplGradX_.cols <= gray.cols && tplGradX_.rows <= gray.rows) {
            // ---- 计算源图 X/Y 方向 Sobel 梯度 ----
            cv::Mat srcGradX, srcGradY;
            computeGradient(gray, srcGradX, srcGradY);

            // ---- X 方向梯度匹配 ----
            // 使用 TM_CCORR_NORMED 而非 TM_CCOEFF_NORMED：
            // CCOEFF 会减去均值，断码白区梯度≈0、方差≈0，
            // 归一化时除以接近零的标准差导致数值爆炸，随机噪声被放大成异常高分
            // CCORR 直接计算余弦相似度，白区低梯度能量自然产生低分
            cv::Mat resultX;
            cv::matchTemplate(srcGradX, tplGradX_, resultX, cv::TM_CCORR_NORMED);

            // ---- Y 方向梯度匹配 ----
            cv::Mat resultY;
            cv::matchTemplate(srcGradY, tplGradY_, resultY, cv::TM_CCORR_NORMED);

            // ---- 合并 X+Y 梯度响应 ----
            // TM_CCORR_NORMED 每通道输出 [0, 1]，求和后除以 2 归一化到 [0, 1]
            cv::Mat result = (resultX + resultY) * 0.5;

            double threshold = th_->getInt() / 100.0;
            int margin = margin_->getInt();
            int tw = tplGradX_.cols;
            int th = tplGradX_.rows;

            // ---- 取全局最高匹配度位置 ----
            double maxVal;
            cv::Point maxLoc;
            cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);
            if (maxVal >= threshold) {
                // ---- 边缘重叠率二次验证 ----
                // 完整匹配：模板边缘与源图边缘高度重叠
                // 断码/残缺：模板在断口处有边缘，源图无对应 → 重叠率暴跌
                bool edgePassed = true;
                if (!tplEdge_.empty()) {
                    cv::Rect matchRoi(maxLoc.x, maxLoc.y, tw, th);
                    cv::Mat srcRoi = gray(matchRoi);
                    cv::Mat srcEdge;
                    cv::Canny(srcRoi, srcEdge, 50, 150);
                    // 膨胀源图边缘 1 像素，容忍 1-2 像素的边缘偏移
                    // Canny 边缘是 1 像素细线，光照/模糊差异导致轻微偏移
                    // 不膨胀时完整条码的重叠率也可能只有 10-30%
                    cv::Mat srcEdgeDilated;
                    cv::dilate(srcEdge, srcEdgeDilated, cv::Mat(), cv::Point(-1,-1), 1);
                    int tplEdgeCount = cv::countNonZero(tplEdge_);
                    if (tplEdgeCount > 0) {
                        int overlapCount = cv::countNonZero(tplEdge_ & srcEdgeDilated);
                        double edgeRatio = (double)overlapCount / tplEdgeCount;
                        int edgeThresh = edgeOverlap_->getInt();
                        edgePassed = (edgeRatio * 100.0 >= edgeThresh);
                    }
                }

                if (edgePassed) {
                    QPolygon poly;
                    poly << QPoint(maxLoc.x - margin, maxLoc.y - margin)
                         << QPoint(maxLoc.x + tw + margin, maxLoc.y - margin)
                         << QPoint(maxLoc.x + tw + margin, maxLoc.y + th + margin)
                         << QPoint(maxLoc.x - margin, maxLoc.y + th + margin);
                    matches.append(poly);
                }
            }

        } else {
            image_[0]->hide();
        }

        // ---- 输出结果 ----
        objList_->setPolygonList(matches);
        diffList_->setPolygonList(QVector<QPolygon>());

        QPolygon searchRegion;
        searchRegion << QPoint(0, 0) << QPoint(gray.cols, 0)
                     << QPoint(gray.cols, gray.rows) << QPoint(0, gray.rows);
        region_->setPolygon(searchRegion);

        setStatus(matches.isEmpty() ? Error : Ok);
    }

private:
    EBoolProperty* first_;
    EIntProperty* th_;
    EIntProperty* margin_;
    EIntProperty* edgeOverlap_;
    ERectProperty* roi_;

    EPolygonListProperty* objList_;
    EPolygonListProperty* diffList_;
    EPolygonProperty* region_;
    EImageProperty* image_[1];

    cv::Mat tplGradX_, tplGradY_;
    cv::Mat tplEdge_;
    bool mbTemplateReady_;
};

// ============================================================
//  DLL 导出接口
// ============================================================
extern "C" _declspec(dllexport) EModule* create(const QString& name, EProperty* parent) {
    return new TemplateMatch(name, parent);
}

extern "C" _declspec(dllexport) const char* getModuleName() {
    return "TemplateMatch";
}

#include "TemplateMatch.moc"
