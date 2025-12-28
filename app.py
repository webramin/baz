import json
import os
import re

def load_json_file(file_path):
    """بارگذاری فایل JSON با کنترل خطا"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"فایل {file_path} یافت نشد!")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"خطا در فرمت JSON: {e}")
    except Exception as e:
        raise Exception(f"خطای ناشناخته: {e}")

def detect_gender_from_name(name, male_names, female_names, male_suffixes, female_indicators):
    """تشخیص جنسیت از نام"""
    if not name or not name.strip():
        return None
    
    name = name.strip()
    name_lower = name.lower()
    
    # نام‌های خاص که استثنا هستند
    special_cases = {
        # نام‌های مردانه که ممکن است شبیه زنانه باشند
        'ترانه': 'مرد',  # ترانه رضوی (بازیگر مرد)
        'افسانه': 'زن',   # معمولاً زنانه است
        'افسر': 'زن',     # معمولاً زنانه است
    }
    
    if name in special_cases:
        return special_cases[name]
    
    # تقسیم نام به اجزاء
    parts = name.split()
    first_part = parts[0] if parts else ''
    last_part = parts[-1] if len(parts) > 1 else ''
    
    # بررسی بر اساس نام کوچک
    if first_part in male_names:
        return 'مرد'
    if first_part in female_names:
        return 'زن'
    
    # بررسی پسوندهای نام خانوادگی
    for suffix in male_suffixes:
        if last_part.endswith(suffix):
            return 'مرد'
    
    # بررسی نشانگرهای زنانه در کل نام
    for indicator in female_indicators:
        if indicator in name_lower:
            return 'زن'
    
    # الگوهای خاص
    patterns = {
        # پسوندهای مردانه در نام خانوادگی
        r'.*پور$': 'مرد',
        r'.*زاده$': 'مرد',
        r'.*نیا$': 'مرد',
        r'.*فر$': 'مرد',
        # پیشوندهای خاص
        r'آقای\s+.*': 'مرد',
        r'خانم\s+.*': 'زن',
        r'دکتر\s+.*': None,  # نامشخص
        r'پروفسور\s+.*': None,  # نامشخص
    }
    
    for pattern, gender in patterns.items():
        if re.match(pattern, name):
            return gender
    
    return None

def main():
    # مسیر فایل
    input_file = 'baz.json'
    output_file = 'actors_processed.json'
    
    try:
        # بارگذاری داده‌ها
        print(f"📂 در حال خواندن فایل {input_file}...")
        actors = load_json_file(input_file)
        print(f"✅ {len(actors)} رکورد خوانده شد.")
        
        # تعریف مجموعه‌های نام (می‌توانید این‌ها را در فایل جداگانه ذخیره کنید)
        male_names = {
            'آرش', 'آرمان', 'ابراهیم', 'احمد', 'اردشیر', 'اسماعیل',
            'امیر', 'امین', 'بهرام', 'بهروز', 'بهمن', 'پارسا',
            'پرویز', 'پژمان', 'جعفر', 'جمشید', 'حسین', 'رضا',
            'سعید', 'علی', 'محمد', 'مهدی', 'مهران', 'نادر', 'هادی'
        }
        
        female_names = {
            'آزاده', 'آزیتا', 'آناهیتا', 'آیدا', 'الهام', 'باران',
            'بهاره', 'بهنوش', 'بیتا', 'پروین', 'ترانه', 'ثریا',
            'حانیه', 'رها', 'سارا', 'شبنم', 'فاطمه', 'مریم',
            'مهدیس', 'نرگس', 'نیلوفر', 'نیوشا'
        }
        
        male_suffixes = {'پور', 'زاده', 'نیا', 'فر', 'لو', 'بیگی'}
        female_indicators = {'خانم', 'بانو'}
        
        # پردازش بازیگران
        print("🔍 در حال تشخیص جنسیت...")
        processed_count = 0
        for actor in actors:
            original_gender = actor.get('gender', '').strip()
            
            # اگر جنسیت از قبل مشخص نیست، تشخیص بده
            if not original_gender:
                name = actor.get('name', '').strip()
                detected_gender = detect_gender_from_name(
                    name, male_names, female_names, male_suffixes, female_indicators
                )
                if detected_gender:
                    actor['gender'] = detected_gender
                    processed_count += 1
            
            # حذف فیلد description اگر وجود دارد
            if 'description' in actor:
                del actor['description']
        
        print(f"✅ جنسیت برای {processed_count} رکورد تشخیص داده شد.")
        
        # ذخیره نتایج
        print(f"💾 در حال ذخیره در {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(actors, f, ensure_ascii=False, indent=4)
        
        # نمایش آمار
        stats = {
            'مرد': 0,
            'زن': 0,
            'نامشخص': 0,
            'کل': len(actors)
        }
        
        for actor in actors:
            gender = actor.get('gender', 'نامشخص')
            if gender in stats:
                stats[gender] += 1
            else:
                stats['نامشخص'] += 1
        
        print("\n📊 آمار نهایی:")
        for key, value in stats.items():
            if key != 'کل':
                percentage = (value / stats['کل']) * 100
                print(f"  {key}: {value} ({percentage:.1f}%)")
        
        print(f"\n🎯 پردازش با موفقیت انجام شد!")
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
