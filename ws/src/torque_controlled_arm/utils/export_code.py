import os
import sys
from datetime import datetime

def gather_code_files(target_dir=None):
    # Determine target directory
    if target_dir is None:
        target_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"[INFO] No directory given. Using current script directory: {target_dir}")
    else:
        target_dir = os.path.abspath(target_dir)
        if not os.path.isdir(target_dir):
            print(f"[ERROR] Provided path is not a valid directory: {target_dir}")
            return

    output_file = os.path.join(target_dir, f"code_export.txt")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# CODE EXPORT - {datetime.now().strftime('%A, %B %d %Y at %H:%M')}\n")
        outfile.write("#" * 80 + "\n\n")

        for filename in sorted(os.listdir(target_dir)):
            if filename.endswith('.py'):
                filepath = os.path.join(target_dir, filename)

                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write("=" * 80 + "\n")
                        outfile.write(f"FILE: {filename}\n")
                        # outfile.write(f"# PATH: {filepath}\n")
                        outfile.write("#" * 80 + "\n\n")

                        outfile.write(infile.read())
                        outfile.write("\n" + "=" * 80 + "\n\n")

                except Exception as e:
                    outfile.write(f"# ERROR READING {filename}: {str(e)}\n\n")

    print(f"[DONE] Code export complete: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        gather_code_files(sys.argv[1])
    else:
        gather_code_files()
