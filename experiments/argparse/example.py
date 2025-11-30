import argparse


def main():
    # 1. パーサーの作成
    # description: ヘルプの冒頭に表示される説明
    # epilog: ヘルプの最後に表示される補足
    parser = argparse.ArgumentParser(
        description='【共有用】argparse機能網羅サンプルスクリプト',
        epilog='使い方で迷ったら -h をつけて実行してください。'
    )

    # --- A. 基本的な位置引数 (必須) ---
    parser.add_argument('text', help='処理したいメインのテキスト文字列')

    # --- B. オプション引数: 型指定とデフォルト値 ---
    # type=int: 整数に変換。文字を入れると自動エラー。
    # default=1: 指定がなければ 1 が入る。
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='テキストを繰り返す回数 (デフォルト: 1)')

    # --- C. フラグ (スイッチ) ---
    # action='store_true': 指定すると True、しないと False になる。値は取らない。
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='詳細なログを表示するモード')

    # --- D. 選択肢の制限 ---
    # choices=[...]: リストの中にある値しか受け付けない。
    parser.add_argument('--mode', choices=['normal', 'upper', 'lower'], default='normal',
                        help='テキスト変換モード (normal, upper, lower)')

    # --- E. 複数の値をリストで受け取る ---
    # nargs='+': 1個以上の引数をリストにする (例: --tags work urgent)
    parser.add_argument('--tags', nargs='+', default=[],
                        help='タグを複数指定可能')

    # --- F. ファイル操作 (書き込み) ---
    # type=argparse.FileType('w'): 自動でファイルを開く。
    # 指定しないと標準出力(コンソール)には出さない仕様にするため、defaultはNone
    parser.add_argument('--out', type=argparse.FileType('w', encoding='utf-8'),
                        help='結果を保存するファイルパス (指定がない場合は保存しない)')

    # 2. 解析の実行 (ここでコマンドライン引数が読み込まれる)
    args = parser.parse_args()

    # ==============================
    # ここから実際の処理ロジック
    # ==============================

    # ログ表示 (フラグの確認)
    if args.verbose:
        print("[INFO] 処理を開始します...")
        print(f"[INFO] 受け取ったデータ: {args}")

    # モードによるテキスト加工 (choicesの利用)
    content = args.text
    if args.mode == 'upper':
        content = content.upper()
    elif args.mode == 'lower':
        content = content.lower()

    # タグの整形 (リストの利用)
    tag_str = ""
    if args.tags:
        # リストをカンマ区切り文字列に
        tag_str = f" [Tags: {', '.join(args.tags)}]"

    # メイン処理 (繰り返しの利用)
    result_lines = []
    for i in range(args.repeat):
        line = f"{i+1}: {content}{tag_str}"
        result_lines.append(line)

    final_output = "\n".join(result_lines)

    # 結果の出力
    print("--- 実行結果 ---")
    print(final_output)

    # ファイルへの保存 (FileTypeの利用)
    if args.out:
        args.out.write(final_output + "\n")
        print(f"\n[Save] 結果を '{args.out.name}' に保存しました。")
        args.out.close()


if __name__ == '__main__':
    main()
