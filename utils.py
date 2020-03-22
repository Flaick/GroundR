import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.models as models

def seq2phrase(seq, idx2word):
    return "_".join([idx2word[idx] for idx in seq[1:-1]])

def phrase2seq(phrase, word2idx):
    return [word2idx[word] for word in phrase.split(' ')]

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to
    """
    with open(fn, 'r', encoding='utf-8') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index': index,
                                             'phrase': phrase,
                                             'phrase_id': p_id,
                                             'phrase_type': p_type})

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def get_test_vis_feature(coor_list_name='./demo/test.npz.npy', test_img_name='./demo/test.jpg', save_path='./demo'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    resnet = models.resnet101(pretrained=True)
    resnet.eval()
    resnet.cuda()

    if not os.path.exists(os.path.join(save_path,'crop/')): os.mkdir(os.path.join(save_path,'crop/z'))
    crop_img_save_path = os.path.join(save_path,'crop/')

    coor_list = np.load(coor_list_name)
    vis_matrix = np.zeros((len(coor_list), 1000), dtype=float)

    for id, crop in enumerate(coor_list):
        img = Image.open(test_img_name)
        crop_img = img.crop(crop)
        crop_img.save(crop_img_save_path+str(id)+'.png')
        # Extract the feature from cropped images
        crop_img = transform(crop_img)
        vis_feature = resnet(crop_img.unsqueeze(0).cuda())
        vis_matrix[id, :] = vis_feature.cpu().detach().numpy()
    return vis_matrix
    # np.save(save_path + 'feature', vis_matrix)

if __name__=='__main__':
    get_test_vis_feature()
    # #############################################################
    # # EXTRACT IMAGE FEATURES  (RESNET101)                       #
    # #############################################################
    # # Build id to idx dictionary
    # id2idx = {id: idx for (idx, id) in enumerate(list(annotationData['boxes'].keys()))}
    #
    # # LOOP THROUGH ALL BOXES
    # objects = list(annotationData['boxes'].keys())
    # vis_matrix = np.zeros((len(objects), 1000), dtype=float)
    # for id in objects:
    #     img = Image.open(data_folder + 'data/flickr30k-images/' + img_id + '.jpg')
    #     # For each Object: extract boxes and unify them
    #     boxes = annotationData['boxes'][id]
    #     box = unify_boxes(boxes) if len(boxes) > 1 else boxes[0]
    #     # For each box: crop original img to box, resize crop to 224z224, normalise image
    #     box_img = crop_and_resize(img, box, (224, 224))
    #     box_img = transform(box_img)
    #     box_img = box_img.unsqueeze(0)
    #     box_img = box_img.cuda()
    #     # Feed image to ResNet-101 and add to visual feature matrix
    #
    #     vis_matrix[id2idx[id], :] = vis_feature.cpu().detach().numpy()
    #
    # np.save('./visualfeatures_data/' + img_id, vis_matrix)
    # with open('./id2idx_data/' + img_id + '.pkl', 'wb') as f:
    #     pickle.dump(id2idx, f)