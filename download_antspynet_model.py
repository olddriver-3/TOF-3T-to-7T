import os
import requests
from urllib.parse import urlparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_file(url, filename, download_folder='downloads'):
    """
    下载单个文件
    """
    # 创建下载文件夹
    os.makedirs(download_folder, exist_ok=True)
    
    # 完整的文件路径
    file_path = os.path.join(download_folder, filename)
    
    try:
        # 发送GET请求
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        
        # 写入文件
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        print(f"✓ 成功下载: {filename}")
        return True
        
    except Exception as e:
        print(f"✗ 下载失败 {filename}: {str(e)}")
        return False

def get_filename_from_url(url):
    """
    从URL中提取文件名
    """
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # 如果URL中没有文件名，使用默认名称
    if not filename or '.' not in filename:
        filename = f"file_{hash(url)}.download"
    
    return filename

def download_all_files(url_dict, download_folder='downloads', max_workers=5):
    """
    下载字典中的所有文件
    """
    print(f"开始下载 {len(url_dict)} 个文件...")
    print(f"下载文件夹: {download_folder}")
    
    # 创建下载文件夹
    os.makedirs(download_folder, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        future_to_filename = {}
        for key, url in url_dict.items():
            filename = f"{key}_{get_filename_from_url(url)}"
            future = executor.submit(download_file, url, filename, download_folder)
            future_to_filename[future] = filename
        
        # 处理完成的任务
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"✗ 任务异常 {filename}: {str(e)}")
                failed_count += 1
    
    print(f"\n下载完成!")
    print(f"成功: {success_count}, 失败: {failed_count}, 总计: {len(url_dict)}")

# 使用示例
if __name__ == "__main__":
    # 示例字典数据
    url_dict = {
        "antsxnetWmh": "https://figshare.com/ndownloader/files/42301059",
        "antsxnetWmhOr": "https://figshare.com/ndownloader/files/42301056",
        "arterialLesionWeibinShi": "https://figshare.com/ndownloader/files/31624922",
        "brainAgeGender": "https://ndownloader.figshare.com/files/22179948",
        "brainAgeDeepBrainNet": "https://ndownloader.figshare.com/files/23573402",
        "brainExtraction": "https://ndownloader.figshare.com/files/22944632",
        "brainExtractionT1": "https://ndownloader.figshare.com/files/27334370",
        "brainExtractionT1v1": "https://ndownloader.figshare.com/files/28057626",
        "brainExtractionRobustT1": "https://figshare.com/ndownloader/files/34821874",
        "brainExtractionT2": "https://ndownloader.figshare.com/files/23066153",
        "brainExtractionRobustT2": "https://figshare.com/ndownloader/files/34870416",
        "brainExtractionRobustT2Star": "https://figshare.com/ndownloader/files/34870413",
        "brainExtractionFLAIR": "https://ndownloader.figshare.com/files/23562194",
        "brainExtractionRobustFLAIR": "https://figshare.com/ndownloader/files/34870407",
        "brainExtractionBOLD": "https://ndownloader.figshare.com/files/22761977",
        "brainExtractionRobustBOLD": "https://figshare.com/ndownloader/files/34870404",
        "brainExtractionFA": "https://ndownloader.figshare.com/files/22761926",
        "brainExtractionMra": "https://figshare.com/ndownloader/files/46335052",
        "brainExtractionRobustFA": "https://figshare.com/ndownloader/files/34870410",
        "brainExtractionNoBrainer": "https://ndownloader.figshare.com/files/22598039",
        "brainExtractionInfantT1T2": "https://ndownloader.figshare.com/files/22968833",
        "brainExtractionInfantT1": "https://ndownloader.figshare.com/files/22968836",
        "brainExtractionInfantT2": "https://ndownloader.figshare.com/files/22968830",
        "brainExtractionBrainWeb20" : "https://figshare.com/ndownloader/files/50685390",
        "brainExtractionT1Hemi" : "https://figshare.com/ndownloader/files/52184666",
        "brainExtractionT1Lobes" : "https://figshare.com/ndownloader/files/52184660",
        "brainSegmentation": "https://ndownloader.figshare.com/files/13900010",
        "brainSegmentationPatchBased": "https://ndownloader.figshare.com/files/14249717",
        "bratsStage1": "https://figshare.com/ndownloader/files/42384756",
        "bratsStage2": "https://figshare.com/ndownloader/files/42685150", 
        "cerebellumWhole": "https://figshare.com/ndownloader/files/41460447",
        "cerebellumTissue": "https://figshare.com/ndownloader/files/41107724",
        "cerebellumLabels": "https://figshare.com/ndownloader/files/41168678",
        "claustrum_axial_0": "https://ndownloader.figshare.com/files/27844068",
        "claustrum_axial_1": "https://ndownloader.figshare.com/files/27844059",
        "claustrum_axial_2": "https://ndownloader.figshare.com/files/27844062",
        "claustrum_coronal_0": "https://ndownloader.figshare.com/files/27844074",
        "claustrum_coronal_1": "https://ndownloader.figshare.com/files/27844071",
        "claustrum_coronal_2": "https://ndownloader.figshare.com/files/27844065",
        "ctHumanLung": "https://ndownloader.figshare.com/files/20005217",
        "deepFlashLeftT1": "https://ndownloader.figshare.com/files/28966269",
        "deepFlashRightT1": "https://ndownloader.figshare.com/files/28966266",
        "deepFlashLeftBoth": "https://ndownloader.figshare.com/files/28966275",
        "deepFlashRightBoth": "https://ndownloader.figshare.com/files/28966272",
        "deepFlashLeftT1Hierarchical": "https://figshare.com/ndownloader/files/31226449",
        "deepFlashRightT1Hierarchical": "https://figshare.com/ndownloader/files/31226452",
        "deepFlashLeftBothHierarchical": "https://figshare.com/ndownloader/files/31226458",
        "deepFlashRightBothHierarchical": "https://figshare.com/ndownloader/files/31226455",
        "deepFlashLeftT1Hierarchical_ri": "https://figshare.com/ndownloader/files/33198794",
        "deepFlashRightT1Hierarchical_ri": "https://figshare.com/ndownloader/files/33198800",
        "deepFlashLeftBothHierarchical_ri": "https://figshare.com/ndownloader/files/33198803",
        "deepFlashRightBothHierarchical_ri": "https://figshare.com/ndownloader/files/33198809",
        "deepFlash2LeftT1Hierarchical": "https://figshare.com/ndownloader/files/46461418",
        "deepFlash2RightT1Hierarchical": "https://figshare.com/ndownloader/files/46461421",
        "deepFlash": "https://ndownloader.figshare.com/files/22933757",
        "deepFlashLeft8": "https://ndownloader.figshare.com/files/25441007",
        "deepFlashRight8": "https://ndownloader.figshare.com/files/25441004",
        "deepFlashLeft16": "https://ndownloader.figshare.com/files/25465844",
        "deepFlashRight16": "https://ndownloader.figshare.com/files/25465847",
        "deepFlashLeft16new": "https://ndownloader.figshare.com/files/25991681",
        "deepFlashRight16new": "https://ndownloader.figshare.com/files/25991678",
        "denoising": "https://ndownloader.figshare.com/files/14235296",
        "dktInner": "https://ndownloader.figshare.com/files/23266943",
        "dktOuter": "https://ndownloader.figshare.com/files/23765132",
        "dktOuterWithSpatialPriors": "https://ndownloader.figshare.com/files/24230768",
        "DesikanKillianyTourvilleOuter": "https://figshare.com/ndownloader/files/52243655",
        "HarvardOxfordAtlasSubcortical": "https://figshare.com/ndownloader/files/51755546",
        "elBicho": "https://ndownloader.figshare.com/files/26736779",
        "lesion_patch": "https://figshare.com/ndownloader/files/43882539",
        "lesion_whole_brain": "https://figshare.com/ndownloader/files/44032017",  # 44162156
        "lesion_flip_brain": "https://figshare.com/ndownloader/files/44032014",
        "lesion_flip_template_brain": "https://figshare.com/ndownloader/files/44032011",
        "ex5_coronal_weights": "https://figshare.com/ndownloader/files/42434193",
        "ex5_sagittal_weights": "https://figshare.com/ndownloader/files/42434202",
        "allen_brain_mask_weights" : "https://figshare.com/ndownloader/files/36999880",  #https://figshare.com/ndownloader/files/42481248
        "allen_brain_leftright_coronal_mask_weights" : "",
        "allen_cerebellum_coronal_mask_weights" : "",
        "allen_cerebellum_sagittal_mask_weights" : "",
        "allen_sr_weights" : "",
        "mouseMriBrainExtraction" : "https://figshare.com/ndownloader/files/44714947",
        "mouseT2wBrainExtraction3D" : "https://figshare.com/ndownloader/files/49188910",
        "mouseT2wBrainParcellation3DNick" : "https://figshare.com/ndownloader/files/44714944",
        "mouseT2wBrainParcellation3DTct" : "https://figshare.com/ndownloader/files/47214538",
        "mouseSTPTBrainParcellation3DJay" : "https://figshare.com/ndownloader/files/46710592",
        "pvs_shiva_t1_0" : "https://figshare.com/ndownloader/files/48660169",
        "pvs_shiva_t1_1" : "https://figshare.com/ndownloader/files/48660193",
        "pvs_shiva_t1_2" : "https://figshare.com/ndownloader/files/48660199",
        "pvs_shiva_t1_3" : "https://figshare.com/ndownloader/files/48660178",
        "pvs_shiva_t1_4" : "https://figshare.com/ndownloader/files/48660172",
        "pvs_shiva_t1_5" : "https://figshare.com/ndownloader/files/48660187",
        "pvs_shiva_t1_flair_0" : "https://figshare.com/ndownloader/files/48660181",
        "pvs_shiva_t1_flair_1" : "https://figshare.com/ndownloader/files/48660175",
        "pvs_shiva_t1_flair_2" : "https://figshare.com/ndownloader/files/48660184",
        "pvs_shiva_t1_flair_3" : "https://figshare.com/ndownloader/files/48660190",
        "pvs_shiva_t1_flair_4" : "https://figshare.com/ndownloader/files/48660196",
        "wmh_shiva_flair_0" : "https://figshare.com/ndownloader/files/48660487",
        "wmh_shiva_flair_1" : "https://figshare.com/ndownloader/files/48660496",
        "wmh_shiva_flair_2" : "https://figshare.com/ndownloader/files/48660493",
        "wmh_shiva_flair_3" : "https://figshare.com/ndownloader/files/48660490",
        "wmh_shiva_flair_4" : "https://figshare.com/ndownloader/files/48660511",
        "wmh_shiva_t1_flair_0" : "https://figshare.com/ndownloader/files/48660529",
        "wmh_shiva_t1_flair_1" : "https://figshare.com/ndownloader/files/48660547",
        "wmh_shiva_t1_flair_2" : "https://figshare.com/ndownloader/files/48660499",
        "wmh_shiva_t1_flair_3" : "https://figshare.com/ndownloader/files/48660550",
        "wmh_shiva_t1_flair_4" : "https://figshare.com/ndownloader/files/48660544",
        "functionalLungMri": "https://ndownloader.figshare.com/files/13824167",
        "hippMapp3rInitial": "https://ndownloader.figshare.com/files/18068408",
        "hippMapp3rRefine": "https://ndownloader.figshare.com/files/18068411",
        "hyperMapp3r": "https://figshare.com/ndownloader/files/38790702",
        "hypothalamus": "https://ndownloader.figshare.com/files/28344378",
        "inpainting_sagittal_rmnet_weights" : "https://figshare.com/ndownloader/files/44367188",
        "inpainting_coronal_rmnet_weights" : "https://figshare.com/ndownloader/files/44294099",
        "inpainting_axial_rmnet_weights" : "https://figshare.com/ndownloader/files/44244446",
        "inpainting_sagittal_rmnet_flair_weights" : "https://figshare.com/ndownloader/files/44511356",
        "inpainting_coronal_rmnet_flair_weights" : "https://figshare.com/ndownloader/files/44468984",
        "inpainting_axial_rmnet_flair_weights" : "https://figshare.com/ndownloader/files/44406923",
        "koniqMBCS": "https://ndownloader.figshare.com/files/24967376",
        "koniqMS": "https://figshare.com/ndownloader/files/35295403",
        "koniqMS2": "https://figshare.com/ndownloader/files/35295397",
        "koniqMS3": "https://ndownloader.figshare.com/files/25474847",
        "lungCtWithPriorsSegmentationWeights": "https://ndownloader.figshare.com/files/28357818",
        "maskLobes": "https://figshare.com/ndownloader/files/30678458",
        "mriSuperResolution": "https://figshare.com/ndownloader/files/35290684",
        "mriModalityClassification": "https://figshare.com/ndownloader/files/41691681",
        "mraVesselWeights_160": "https://figshare.com/ndownloader/files/46406029",
        "protonLungMri": "https://ndownloader.figshare.com/files/13606799",
        "protonLobes": "https://figshare.com/ndownloader/files/30678455",
        "pulmonaryArteryWeights": "https://figshare.com/ndownloader/files/46400752",
        "pulmonaryAirwayWeights": "https://figshare.com/ndownloader/files/45187168",
        "sixTissueOctantBrainSegmentation": "https://ndownloader.figshare.com/files/23776025",
        "sixTissueOctantBrainSegmentationWithPriors1": "https://ndownloader.figshare.com/files/28159869",
        "sysuMediaWmhFlairOnlyModel0": "https://ndownloader.figshare.com/files/22898441",
        "sysuMediaWmhFlairOnlyModel1": "https://ndownloader.figshare.com/files/22898570",
        "sysuMediaWmhFlairOnlyModel2": "https://ndownloader.figshare.com/files/22898438",
        "sysuMediaWmhFlairT1Model0": "https://ndownloader.figshare.com/files/22898450",
        "sysuMediaWmhFlairT1Model1": "https://ndownloader.figshare.com/files/22898453",
        "sysuMediaWmhFlairT1Model2": "https://ndownloader.figshare.com/files/22898459",
        "tidsQualityAssessment": "https://figshare.com/ndownloader/files/35295391",
        "xrayLungExtraction": "https://figshare.com/ndownloader/files/41965818",
        "xrayLungOrientation": "https://figshare.com/ndownloader/files/41965821",
        "chexnetClassification": "https://figshare.com/ndownloader/files/42423522",
        "chexnetANTsXNetClassification": "https://figshare.com/ndownloader/files/42428943",
        "tb_antsxnet": "https://figshare.com/ndownloader/files/45820599",
        "wholeTumorSegmentationT2Flair": "https://ndownloader.figshare.com/files/14087045",
        "wholeLungMaskFromVentilation": "https://ndownloader.figshare.com/files/28914441",
        "DeepAtroposHcpT1Weights": "https://figshare.com/ndownloader/files/51906071",
        "DeepAtroposHcpT1T2Weights": "https://figshare.com/ndownloader/files/52392374", 
        "DeepAtroposHcpT1FAWeights": "https://figshare.com/ndownloader/files/52392368",
        "DeepAtroposHcpT1T2FAWeights": "https://figshare.com/ndownloader/files/52392371",
        "sig_smallshort_train_1x1x2_1chan_featgraderL6_best_mdl": "https://figshare.com/ndownloader/files/49339837",
        "sig_smallshort_train_1x1x2_1chan_featvggL6_best_mdl": "https://figshare.com/ndownloader/files/49339840",
        "sig_smallshort_train_1x1x3_1chan_featgraderL6_best_mdl": "https://figshare.com/ndownloader/files/49339843",
        "sig_smallshort_train_1x1x3_1chan_featvggL6_best_mdl": "https://figshare.com/ndownloader/files/49339846",
        "sig_smallshort_train_1x1x4_1chan_featgraderL6_best_mdl": "https://figshare.com/ndownloader/files/49339849",
        "sig_smallshort_train_1x1x4_1chan_featvggL6_best_mdl": "https://figshare.com/ndownloader/files/49339852",
        # "sig_smallshort_train_1x1x6_1chan_featgraderL6_best_mdl", Not available
        "sig_smallshort_train_1x1x6_1chan_featvggL6_best_mdl": "https://figshare.com/ndownloader/files/49339855",
        "sig_smallshort_train_2x2x2_1chan_featgraderL6_best_mdl": "https://figshare.com/ndownloader/files/49339858",
        "sig_smallshort_train_2x2x2_1chan_featvggL6_best_mdl": "https://figshare.com/ndownloader/files/49339861",
        "sig_smallshort_train_2x2x4_1chan_featgraderL6_best_mdl": "https://figshare.com/ndownloader/files/49339867",
        "sig_smallshort_train_2x2x4_1chan_featvggL6_best_mdl": "https://figshare.com/ndownloader/files/49339864"            
    }

    # 下载所有文件
    download_all_files(
        url_dict=url_dict,
        download_folder="my_downloads",  # 自定义下载文件夹
        max_workers=3  # 并发下载数量
    )
