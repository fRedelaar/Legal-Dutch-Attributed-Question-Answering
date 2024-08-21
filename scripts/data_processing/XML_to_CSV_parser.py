import os
import xml.etree.ElementTree as ET
import csv

def process_text(element, skip_tags=['opmerkingen-inhoud']):
    """
    Process the text of articles correctly, skipping specified tags.

    Parameters:
    element (xml.etree.Element): XML element.
    skip_tags (list): Tags to be skipped. Default is ['opmerkingen-inhoud'].

    Returns:
    str: Processed text.
    """
    if element.tag in skip_tags:
        return ''  # Skip the text for the specified tags
    text = element.text or ''
    for child in element:
        text += process_text(child, skip_tags)
        if child.tail:
            text += child.tail
    text = text.replace('\u00A0', ' ')  # Replace non-breaking spaces with regular spaces
    return text.strip()

def build_parent_map(element):
    """
    Build a parent map for an XML element.

    Parameters:
    element (xml.etree.Element): XML element.

    Returns:
    dict: Parent map.
    """
    parent_map = {c: p for p in element.iter() for c in p}
    return parent_map

def find_closest_section_title(element, section_type, parent_map):
    """
    Find the closest section title for a given element.

    Parameters:
    element (xml.etree.Element): XML element.
    section_type (str): Type of section.
    parent_map (dict): Parent map.

    Returns:
    str: Closest section title.
    """
    parent = parent_map.get(element)
    while parent is not None:
        if parent.tag == section_type:
            title_element = parent.find('.//titel')
            if title_element is not None:
                return process_text(title_element, [])  # Ensure title text is cleaned without skipping any tags
        parent = parent_map.get(parent)
    return None

def find_closest_subparagraaf_title(element, parent_map):
    """
    Find the closest subparagraaf title for a given element.

    Parameters:
    element (xml.etree.Element): XML element.
    parent_map (dict): Parent map.

    Returns:
    str: Closest subparagraaf title.
    """
    parent = parent_map.get(element)
    while parent is not None:
        if parent.tag == 'sub-paragraaf':
            title_element = parent.find('.//titel')
            if title_element is not None:
                return process_text(title_element, [])
        parent = parent_map.get(parent)
    return None

def extract_article_name(artikel, parent_map):
    """
    Extract article name from an 'artikel' element.

    Parameters:
    artikel (xml.etree.Element): 'artikel' element.
    parent_map (dict): Parent map to navigate the XML tree.

    Returns:
    str: Article name if found, otherwise an empty string.
    """
    # Directly find a 'kop' element within 'artikel'
    kop = artikel.find('.//kop')
    if kop is not None:
        titel_element = kop.find('.//titel')
        if titel_element is not None:
            return process_text(titel_element, [])

    # If not found directly within, check if the parent or nearby element has it
    parent = parent_map.get(artikel)
    while parent is not None:
        kop = parent.find('.//kop')
        if kop:
            titel_element = kop.find('.//titel')
            if titel_element:
                return process_text(titel_element, [])
        parent = parent_map.get(parent)

    return ''

def extract_title_title(artikel, parent_map):
    """
    Extract the title name from an 'artikel' element.

    Parameters:
    artikel (xml.etree.Element): 'artikel' element.
    parent_map (dict): Parent map to navigate the XML tree.

    Returns:
    str: Title name if found, otherwise an empty string.
    """
    # Initialize the parent using the parent_map
    parent = parent_map.get(artikel)

    # Traverse upwards through the XML tree
    while parent is not None:
        # Look for a 'kop' element with a 'label' child having text "Titel"
        kop = parent.find('.//kop[label="Titel"]')
        if kop is not None:
            # If found, extract the 'titel' text
            titel_element = kop.find('.//titel')
            if titel_element is not None and titel_element.text:
                return titel_element.text.strip()
        # Move to the next parent in the hierarchy
        parent = parent_map.get(parent)

    return ''

def find_elements_within_artikels(element, tag_name, skip_tags=['opmerkingen-inhoud']):
    """
    Find elements within 'artikel' tags while skipping specified tags.

    Parameters:
    element (xml.etree.Element): XML element.
    tag_name (str): Name of the tag to find.
    skip_tags (list): Tags to be skipped. Default is ['opmerkingen-inhoud'].

    Returns:
    list: List of found elements.
    """
    elements = []
    if element.tag in skip_tags:
        return elements  # Return empty list if the current element should be skipped
    for child in element:
        if child.tag == tag_name:
            elements.append(child)
        else:
            # Extend the list with elements found in child, skipping specified tags
            elements.extend(find_elements_within_artikels(child, tag_name, skip_tags))
    return elements

def extract_section_identifiers(artikel):
    """
    Extract section identifiers from an 'artikel' element.

    Parameters:
    artikel (xml.etree.Element): 'artikel' element.

    Returns:
    tuple: Tuple containing hoofdstuk, afdeling, and paragraaf identifiers.
    """
    path_elements = artikel.attrib.get('bwb-ng-variabel-deel', 'no-bwb').split('/')
    hoofdstuk_nr, afdeling_nr, paragraaf_nr = '', '', ''
    if len(path_elements) > 1:
        hoofdstuk_nr = path_elements[1]
    if len(path_elements) > 2:
        afdeling_nr = path_elements[2]
    if len(path_elements) > 3:
        paragraaf_nr = path_elements[3]
    return hoofdstuk_nr, afdeling_nr, paragraaf_nr

def process_and_write_csv(file_path, csv_file_path, law_id, law_name):
    """
    Process XML file, extract data, and write to a CSV file.

    Parameters:
    file_path (str): Path to the XML file.
    csv_file_path (str): Path to the output CSV file.
    law_id (str): Law ID.
    law_name (str): Law name.

    Returns:
    None
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    parent_map = build_parent_map(root)
    artikels = root.findall(".//artikel")

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['law_id', 'law_name', 'hoofdstuk', 'hoofdstuk_titel', 'afdeling', 'afdeling_titel', 'paragraaf',
             'paragraaf_titel', 'subparagraaf_titel', 'titel_titel', 'artikel', 'article_name', 'text'])

        for artikel in artikels:
            hoofdstuk_nr, afdeling_nr, paragraaf_nr = extract_section_identifiers(artikel)

            hoofdstuk_titel = find_closest_section_title(artikel, 'hoofdstuk', parent_map)
            afdeling_titel = find_closest_section_title(artikel, 'afdeling', parent_map)
            paragraaf_titel = find_closest_section_title(artikel, 'paragraaf', parent_map)
            subparagraaf_titel = find_closest_subparagraaf_title(artikel, parent_map)
            article_name = extract_article_name(artikel, parent_map)
            titel_titel = extract_title_title(artikel, parent_map)

            artikel_label = artikel.attrib.get('label', 'nolabel')

            al_elements = find_elements_within_artikels(artikel, 'al')
            artikel_text = ' '.join([process_text(al_element) for al_element in al_elements])

            writer.writerow([law_id, law_name, hoofdstuk_nr, hoofdstuk_titel, afdeling_nr, afdeling_titel, paragraaf_nr,
                             paragraaf_titel, subparagraaf_titel, titel_titel, artikel_label, article_name,
                             artikel_text])

def extract_law_id(file_path):
    """
    Extract the law ID from the XML file.

    Parameters:
    file_path (str): Path to the XML file.

    Returns:
    str: Law ID if found, otherwise 'Unknown'.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    citeertitel_element = root.find('.//citeertitel')
    if citeertitel_element is not None and citeertitel_element.text:
        return citeertitel_element.text.strip()
    return 'Unknown'

def main():
    input_dir = r"../../datasets/knowledge_corpus/large_corpus/law_articles_XML_format"
    output_dir = r"../../datasets/knowledge_corpus/large_corpus/law_articles_CSV_format"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(input_dir, filename)
            csv_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.csv')
            law_id = filename.split('_')[0]
            law_name = extract_law_id(file_path)
            process_and_write_csv(file_path, csv_file_path, law_id, law_name)

if __name__ == "__main__":
    main()
