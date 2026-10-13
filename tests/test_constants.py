"""Weight-free tests for kwja_mlx.constants (table shapes, fidelity to upstream kwja)."""

from kwja_mlx.constants import (
    BASE_PHRASE_FEATURES,
    BR_CASES,
    CHOON_SET,
    COHESION_RELATIONS,
    CONJFORM_TAGS,
    CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID,
    CONJTYPE_TAGS,
    COREFERENCE_RELS,
    DEPENDENCY_TYPES,
    DISCOURSE_RELATIONS,
    EXOPHORA_REFERENTS,
    HATSUON_SET,
    ID,
    ID_ID,
    IGNORE_WORD_NORM_OP_TAG,
    INFLECTABLE,
    LOWER2UPPER,
    PAS_CASES,
    POS_TAG2POS_ID,
    POS_TAG_SUBPOS_TAG2SUBPOS_ID,
    POS_TAGS,
    PROLONGED_MAP,
    PROLONGED_MAP_FOR_EROW,
    READING_VOCAB,
    SENT_SEGMENTATION_TAGS,
    SPECIALS,
    SUBPOS_TAGS,
    UNK,
    UNK_ID,
    VOICED2VOICELESS,
    WORD_FEATURES,
    WORD_NORM_OP_TAGS,
    WORD_SEGMENTATION_TAGS,
    get_reading2reading_id,
)


class TestTagTables:
    """Tag tuples for char-module heads."""

    def test_sentence_segmentation_tags(self):
        assert SENT_SEGMENTATION_TAGS == ("B", "I")

    def test_word_segmentation_tags(self):
        assert WORD_SEGMENTATION_TAGS == ("B", "I")

    def test_word_norm_op_tags(self):
        assert WORD_NORM_OP_TAGS == ("K", "D", "V", "S", "P", "E")

    def test_ignore_word_norm_op_tag(self):
        assert IGNORE_WORD_NORM_OP_TAG == "_"


class TestNormOpRestore:
    """Constants for word-normalization reading restoration."""

    def test_choon_set(self):
        for ch in ("ー", "ｰ", "-", "〜", "⁓", "～", "~", "∼"):
            assert ch in CHOON_SET

    def test_hatsuon_set(self):
        assert HATSUON_SET == frozenset({"っ", "ッ"})

    def test_lower2upper(self):
        assert LOWER2UPPER["ぁ"] == "あ"
        assert LOWER2UPPER["ヶ"] == "ケ"

    def test_voiced2voiceless(self):
        assert len(VOICED2VOICELESS) == 50
        assert VOICED2VOICELESS["が"] == "か"
        assert VOICED2VOICELESS["パ"] == "ハ"

    def test_prolonged_maps(self):
        assert PROLONGED_MAP["か"] == "あ"
        assert PROLONGED_MAP["く"] == "う"
        assert PROLONGED_MAP_FOR_EROW["え"] == "え"


class TestMorphologicalAnalysis:
    """POS/SUBPOS/CONJTYPE/CONJFORM tables."""

    def test_pos_tables(self):
        assert len(POS_TAGS) == 14
        assert POS_TAG2POS_ID["名詞"] == 6
        assert tuple(POS_TAG2POS_ID.keys()) == POS_TAGS

    def test_subpos_tags_length(self):
        assert len(SUBPOS_TAGS) == 35
        assert SUBPOS_TAGS[0] == "*"

    def test_pos_subpos_constraints(self):
        assert set(POS_TAG_SUBPOS_TAG2SUBPOS_ID.keys()) == set(POS_TAGS)
        assert POS_TAG_SUBPOS_TAG2SUBPOS_ID["動詞"] == {"*": 0}
        assert POS_TAG_SUBPOS_TAG2SUBPOS_ID["名詞"]["数詞"] == 7
        # every non-star subpos tag appears in SUBPOS_TAGS
        for subpos in POS_TAG_SUBPOS_TAG2SUBPOS_ID.values():
            for tag in subpos:
                assert tag in SUBPOS_TAGS

    def test_conjtype_tags_length(self):
        assert len(CONJTYPE_TAGS) == 33
        assert CONJTYPE_TAGS[0] == "*"
        assert "サ変動詞" in CONJTYPE_TAGS

    def test_conjform_tags_length(self):
        assert len(CONJFORM_TAGS) == 81
        assert CONJFORM_TAGS[:3] == ("*", "語幹", "基本形")
        assert "文語命令形" in CONJFORM_TAGS

    def test_conjtype_conjform_constraints(self):
        assert set(CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID.keys()) == set(CONJTYPE_TAGS)
        assert CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID["*"] == {"*": 0}
        for conjtype, conjforms in CONJTYPE_TAG_CONJFORM_TAG2CONJFORM_ID.items():
            ids = sorted(conjforms.values())
            assert ids == list(range(len(conjforms))), conjtype
            for conjform in conjforms:
                assert conjform in CONJFORM_TAGS, (conjtype, conjform)

    def test_inflectable(self):
        for pos in ("動詞", "形容詞", "判定詞", "助動詞"):
            for subpos in POS_TAG_SUBPOS_TAG2SUBPOS_ID[pos]:
                assert (pos, subpos) in INFLECTABLE
        assert ("接尾辞", "動詞性接尾辞") in INFLECTABLE
        assert ("名詞", "*") not in INFLECTABLE


class TestWordModuleFeatures:
    """Word / base phrase feature tag tables."""

    def test_word_features(self):
        assert WORD_FEATURES == (
            "基本句-主辞",
            "基本句-区切",
            "文節-区切",
            "用言表記先頭",
            "用言表記末尾",
        )

    def test_base_phrase_features(self):
        assert len(BASE_PHRASE_FEATURES) == 64
        assert BASE_PHRASE_FEATURES[0] == "用言:動"
        assert "節-主辞" in BASE_PHRASE_FEATURES
        assert "モダリティ-疑問" in BASE_PHRASE_FEATURES


class TestDependencyAndCohesion:
    """Dependency types and cohesion rel names."""

    def test_dependency_types(self):
        # rhoknp.props.DepType definition order
        assert DEPENDENCY_TYPES == ("D", "P", "A", "I")

    def test_pas_br_relations(self):
        assert PAS_CASES == ("ガ", "ヲ", "ニ", "ガ２", "デ", "ト", "時間")
        assert BR_CASES == ("ノ",)
        assert COREFERENCE_RELS == ("=",)
        assert len(COHESION_RELATIONS) == 9
        assert COHESION_RELATIONS == PAS_CASES + BR_CASES + COREFERENCE_RELS

    def test_specials(self):
        # configs/word_module.yaml special_tokens minus trailing " "
        assert SPECIALS == (
            "[著者]",
            "[読者]",
            "[不特定:人]",
            "[不特定:物]",
            "[NULL]",
            "[NA]",
            "[ROOT]",
        )

    def test_exophora_referents(self):
        assert EXOPHORA_REFERENTS == ("著者", "読者", "不特定:人", "不特定:物")


class TestDiscourseRelations:
    """Discourse relation labels."""

    def test_discourse_relations(self):
        assert DISCOURSE_RELATIONS == (
            "談話関係なし",
            "原因・理由",
            "目的",
            "条件",
            "根拠",
            "対比",
            "逆接",
        )


class TestReadingPrediction:
    """Reading vocab resource and id mapping."""

    def test_special_tokens(self):
        assert UNK == "[UNK]"
        assert ID == "[ID]"
        assert UNK_ID == 0
        assert ID_ID == 1

    def test_reading_vocab_size(self):
        assert len(READING_VOCAB) == 12906
        assert READING_VOCAB[0] == "の"

    def test_reading2reading_id(self):
        reading2reading_id = get_reading2reading_id()
        assert len(reading2reading_id) == 12907
        assert reading2reading_id[UNK] == 0
        assert reading2reading_id[ID] == 1
        assert reading2reading_id["の"] == 2
        ids = sorted(reading2reading_id.values())
        assert ids == list(range(len(ids)))
